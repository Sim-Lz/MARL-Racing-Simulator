# track.py
import math

import numpy as np
import pygame

from config import (
    ACCELERATION,
    CAR_HALF_WIDTH,
    DECELERATION,
    FPS,
    MAX_SPEED_PX,
    MAX_STEER_ANGLE,
    MIN_SPEED_PX,
    NUM_CHECKPOINTS,
    RACING_LINE_MARGIN_PX,
    RACING_LINE_SPEED_MARGIN,
)

CENTERLINE_STEP_PX = 2.0
CENTERLINE_SMOOTHING_POINTS = 9
PROJECTION_WINDOW_POINTS = 40
RACING_LINE_COARSE_STEP_PX = 10.0
RACING_LINE_MAX_ACTIVE_SET_ITERATIONS = 60

# Limiti di curvatura del modello di sterzo (vedi Car.steering_rate):
#  - sotto MIN_SPEED il raggio minimo è costante: k_max = rad(MAX_STEER) * FPS / MIN_SPEED;
#  - sopra MIN_SPEED la curvatura massima scala come K / v^2.
CURVATURE_CAPACITY = math.radians(MAX_STEER_ANGLE) * MIN_SPEED_PX * FPS
MAX_CURVATURE = math.radians(MAX_STEER_ANGLE) * FPS / MIN_SPEED_PX


def speed_limit_for_curvature(curvature, margin=1.0):
    """Velocità massima a cui il modello di sterzo riesce a percorrere la curvatura data."""
    curvature = np.maximum(np.abs(np.asarray(curvature, dtype=np.float64)), 1e-9)
    limit = np.sqrt(margin * CURVATURE_CAPACITY / curvature)
    return np.clip(limit, MIN_SPEED_PX, MAX_SPEED_PX)


def minimum_curvature_offsets(center, normal, lower, upper, regularization=1e-6):
    """Risolve min ||D2 (c + n * a)||^2 con lower <= a <= upper (minimi quadrati con vincoli di box, active set).

    D2 è la differenza seconda circolare: la somma dei suoi quadrati approssima la curvatura totale al
    quadrato del percorso, quindi la soluzione è la traiettoria più "dritta" possibile entro la pista.
    """
    count = len(center)
    second = np.zeros((count, count))
    rows = np.arange(count)
    second[rows, rows] = -2.0
    second[rows, (rows - 1) % count] = 1.0
    second[rows, (rows + 1) % count] = 1.0
    matrix = np.vstack([second * normal[:, 0], second * normal[:, 1]])
    target = np.concatenate([second @ center[:, 0], second @ center[:, 1]])
    hessian = matrix.T @ matrix + regularization * np.eye(count)
    gradient0 = matrix.T @ target

    offset = np.zeros(count)
    clamped = np.zeros(count, dtype=bool)
    for _ in range(RACING_LINE_MAX_ACTIVE_SET_ITERATIONS):
        free = ~clamped
        if free.any():
            rhs = -(gradient0[free] + hessian[np.ix_(free, clamped)] @ offset[clamped])
            offset[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs)
        violating = free & ((offset > upper) | (offset < lower))
        offset = np.clip(offset, lower, upper)
        gradient = hessian @ offset + gradient0
        # Libera i vincoli attivi il cui gradiente spinge verso l'interno (condizioni KKT).
        release = clamped & (((offset >= upper) & (gradient > 0)) | ((offset <= lower) & (gradient < 0)))
        if not violating.any() and not release.any():
            break
        clamped = (clamped | violating) & ~release
    return offset


def speed_profile(points, margin=RACING_LINE_SPEED_MARGIN, smoothing=7):
    """Profilo di velocità limite lungo un percorso chiuso: (velocità, curvatura, tempo sul giro)."""
    count = len(points)
    segment = np.roll(points, -1, axis=0) - points
    segment_length = np.maximum(np.linalg.norm(segment, axis=1), 1e-6)
    headings = np.arctan2(segment[:, 1], segment[:, 0])
    turn = (headings - np.roll(headings, 1) + np.pi) % (2.0 * np.pi) - np.pi
    curvature = smooth_circular(turn / segment_length, smoothing)

    speed = speed_limit_for_curvature(curvature, margin)
    speed = np.where(np.abs(curvature) > MAX_CURVATURE, MIN_SPEED_PX, speed)
    for _ in range(2):  # due giri per gestire la chiusura dell'anello
        for i in range(count - 1, -1, -1):
            nxt = speed[(i + 1) % count]
            speed[i] = min(speed[i], math.sqrt(nxt * nxt + 2.0 * DECELERATION * margin * segment_length[i]))
        for i in range(count):
            prev = speed[i - 1]
            speed[i] = min(speed[i], math.sqrt(prev * prev + 2.0 * ACCELERATION * segment_length[i - 1]))
    return speed, curvature, float(np.sum(segment_length / np.maximum(speed, 1.0)))


def smooth_circular(values, window):
    kernel = np.ones(window) / window
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


class Track:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.surface.fill((0, 0, 0))
        self.brush_radius = 30

        self.start_position = None
        self.start_angle = 0.0
        self.perimeter = 0.0
        self.drawing_points = []
        self.checkpoints = []

        # Geometria derivata (popolata da finalize)
        self.mask = None
        self.center = None
        self.tangent = None
        self.normal = None
        self.curvature = None
        self.corner_curvature = None
        self.length = 0.0
        self.step = CENTERLINE_STEP_PX
        # Traiettoria ideale (popolata da build_racing_line)
        self.racing_offset = None
        self.racing_points = None
        self.racing_speed = None
        self.racing_curvature = None
        self.ideal_lap_time = float("inf")

    def draw_step(self, screen, mouse_pos=None):
        mouse_pos = pygame.mouse.get_pos() if mouse_pos is None else mouse_pos
        buttons = pygame.mouse.get_pressed()

        if buttons[0]:
            clamped_x = max(0, min(mouse_pos[0], self.width - 1))
            clamped_y = max(0, min(mouse_pos[1], self.height - 1))
            safe_pos = (clamped_x, clamped_y)

            if self.start_position is None:
                self.start_position = safe_pos
                self.drawing_points.append(safe_pos)
            else:
                last_p = self.drawing_points[-1]
                dist = math.hypot(safe_pos[0] - last_p[0], safe_pos[1] - last_p[1])

                if dist > 3:  # Risoluzione di disegno aumentata
                    self.perimeter += dist
                    self.drawing_points.append(safe_pos)

                    if len(self.drawing_points) == 6:
                        p0 = self.drawing_points[0]
                        p5 = self.drawing_points[5]
                        self.start_angle = math.degrees(math.atan2(p5[1] - p0[1], p5[0] - p0[0]))

            pygame.draw.circle(self.surface, (255, 255, 255), safe_pos, self.brush_radius)

        screen.blit(self.surface, (0, 0))
        pygame.draw.circle(screen, (0, 255, 0), mouse_pos, self.brush_radius, 2)

        font = pygame.font.SysFont("Arial", 20)
        text = font.render(f"Disegna la pista. Frecce SU/GIU per Spessore: {self.brush_radius*2}px | Premi INVIO per confermare.", True, (255, 255, 255))
        screen.blit(text, (20, 20))

    def close_track(self):
        """Sana il gap lineare iniziale-finale ed esegue il post-processing di uniformità superficiale."""
        if len(self.drawing_points) < 2:
            return

        p_last = self.drawing_points[-1]
        p_first = self.drawing_points[0]
        dist = math.hypot(p_first[0] - p_last[0], p_first[1] - p_last[1])

        if dist > 2:
            steps = int(dist / 2)
            for i in range(1, steps + 1):
                t = i / steps
                curr_x = int(p_last[0] + (p_first[0] - p_last[0]) * t)
                curr_y = int(p_last[1] + (p_first[1] - p_last[1]) * t)
                self.drawing_points.append((curr_x, curr_y))

        # --- ALGORITMO DI POST-PROCESSING SUPERFICIALE CONTRO I BUCHI ---
        processed_points = []
        for i in range(len(self.drawing_points)):
            p_curr = self.drawing_points[i]
            p_next = self.drawing_points[(i + 1) % len(self.drawing_points)]
            segment_dist = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            processed_points.append(p_curr)
            pygame.draw.circle(self.surface, (255, 255, 255), p_curr, self.brush_radius)

            if segment_dist > 2.0:
                fill_steps = int(segment_dist / 2.0)
                for s in range(1, fill_steps):
                    t = s / fill_steps
                    fx = int(p_curr[0] + (p_next[0] - p_curr[0]) * t)
                    fy = int(p_curr[1] + (p_next[1] - p_curr[1]) * t)
                    processed_points.append((fx, fy))
                    pygame.draw.circle(self.surface, (255, 255, 255), (fx, fy), self.brush_radius)

        self.drawing_points = processed_points

    def finalize(self):
        """Chiude il tracciato e costruisce maschera, linea centrale e checkpoint. Ritorna False se non valido."""
        self.close_track()
        self.build_mask()
        if not self.build_centerline():
            return False
        self.generate_checkpoints()
        self.build_racing_line()
        return True

    def build_mask(self):
        self.mask = pygame.surfarray.array_red(self.surface) > 50
        # Bordo di un pixel non percorribile: evita i controlli di bounds nelle lookup vettoriali.
        self._padded_mask = np.pad(self.mask, 1, constant_values=False)

    def build_centerline(self):
        """Linea centrale levigata e ricampionata a passo costante (ascissa curvilinea uniforme)."""
        points = np.asarray(self.drawing_points, dtype=np.float64)
        if len(points) < 3:
            return False
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.hypot(*(points[1:] - points[:-1]).T) > 0.5
        points = points[keep]
        if len(points) > 2 and math.hypot(*(points[-1] - points[0])) < 0.5:
            points = points[:-1]
        if len(points) < 3:
            return False

        window = min(CENTERLINE_SMOOTHING_POINTS, len(points))
        offsets = np.arange(window) - window // 2
        indices = (np.arange(len(points))[:, None] + offsets[None, :]) % len(points)
        smoothed = points[indices].mean(axis=1)
        # Il punto di partenza resta quello disegnato dall'utente.
        smoothed += points[0] - smoothed[0]

        closed = np.vstack([smoothed, smoothed[:1]])
        segment = np.hypot(*(closed[1:] - closed[:-1]).T)
        arc = np.concatenate([[0.0], np.cumsum(segment)])
        length = float(arc[-1])
        if length < 10 * CENTERLINE_STEP_PX:
            return False

        count = max(3, int(round(length / CENTERLINE_STEP_PX)))
        samples = np.arange(count) * (length / count)
        center = np.column_stack([np.interp(samples, arc, closed[:, 0]), np.interp(samples, arc, closed[:, 1])])

        forward = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
        tangent = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
        headings = np.arctan2(tangent[:, 1], tangent[:, 0])
        turn = (np.roll(headings, -1) - headings + np.pi) % (2.0 * np.pi) - np.pi

        self.length = length
        self.step = length / count
        self.center = center
        self.tangent = tangent
        self.normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # verso destra (asse y verso il basso)
        self.curvature = turn / self.step
        self.corner_curvature = smooth_circular(self.curvature, 11)
        self.drawing_points = [(float(x), float(y)) for x, y in center]
        self.start_angle = math.degrees(headings[0])
        return True

    def build_racing_line(self):
        """Traiettoria a curvatura minima entro i limiti della pista e profilo di velocità limite.

        La traiettoria è espressa come spostamento lungo le normali di una linea di riferimento: la mezzeria
        fortemente levigata (sui tracciati disegnati a mano la mezzeria è ondulata, mentre la zona
        percorribile, unione dei cerchi del pennello, è molto più regolare). I limiti laterali sono misurati
        sulla maschera lungo quelle normali. Gli spostamenti minimizzano la somma delle curvature al
        quadrato: ne risulta la classica linea esterno-interno-esterno che taglia dove la pista lo consente.
        Il profilo di velocità rispetta il limite di curvatura del modello di sterzo, la frenata (passata
        all'indietro) e l'accelerazione (passata in avanti).
        """
        count = len(self.center)
        stride = max(1, int(round(RACING_LINE_COARSE_STEP_PX / self.step)))
        coarse = np.arange(0, count, stride)

        # Linea di riferimento: mezzeria levigata su ~2 larghezze di pista, tenuta in pista.
        window = max(3, int(2.0 * self.brush_radius * 2.0 / self.step) | 1)
        reference = np.column_stack([
            smooth_circular(self.center[:, 0], window), smooth_circular(self.center[:, 1], window)
        ])[coarse]
        off_track = ~self.drivable(reference[:, 0], reference[:, 1])
        reference[off_track] = self.center[coarse][off_track]
        forward = np.roll(reference, -1, axis=0) - np.roll(reference, 1, axis=0)
        tangent = forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        headings = np.arctan2(tangent[:, 1], tangent[:, 0])
        spacing = np.maximum(np.linalg.norm(np.roll(reference, -1, axis=0) - reference, axis=1), 1e-6)
        curvature = smooth_circular(((np.roll(headings, -1) - headings + np.pi) % (2.0 * np.pi) - np.pi) / spacing, 3)

        clearance = CAR_HALF_WIDTH + RACING_LINE_MARGIN_PX
        upper = self._free_distance(reference, normal) - clearance
        lower = -(self._free_distance(reference, -normal) - clearance)
        # Verso l'interno di una curva, oltre il centro di curvatura le normali si incrociano e la traiettoria
        # formerebbe un'ansa impossibile: il limite interno scala con il raggio locale del riferimento.
        inner_limit = 0.8 / np.maximum(np.abs(curvature), 1e-9)
        upper = np.where(curvature > 0, np.minimum(upper, inner_limit), upper)   # curvatura > 0: interno = +normale
        lower = np.where(curvature < 0, np.maximum(lower, -inner_limit), lower)
        upper = np.maximum(upper, 0.0)
        lower = np.minimum(lower, 0.0)

        offset = minimum_curvature_offsets(reference, normal, lower, upper)

        # Punti e profilo si calcolano sulla soluzione a passo largo: ricostruirli a passo fine sulla mezzeria
        # reintrodurrebbe il tremolio del disegno (falsi picchi di curvatura, velocità limite troppo basse).
        coarse_points = reference + normal * offset[:, None]
        coarse_speed, coarse_curvature, self.ideal_lap_time = speed_profile(
            coarse_points, margin=RACING_LINE_SPEED_MARGIN, smoothing=3
        )

        # Interpolazione circolare sugli indici della linea centrale fine.
        fine = np.arange(count)
        extended_index = np.concatenate([coarse, [count]])

        def periodic(values):
            return np.interp(fine, extended_index, np.concatenate([values, values[:1]]))

        self.racing_points = np.column_stack([periodic(coarse_points[:, 0]), periodic(coarse_points[:, 1])])
        self.racing_speed = periodic(coarse_speed)
        self.racing_curvature = periodic(coarse_curvature)
        # Offset della traiettoria e limiti laterali rispetto alla mezzeria (usati da sorpasso e metriche).
        delta = self.racing_points - self.center
        self.racing_offset = np.einsum("ij,ij->i", delta, self.normal)
        # Limitati alla semi-larghezza nominale: su una mezzeria ondulata la distanza libera lungo la normale
        # può attraversare la zona sovrapposta di un tornante e finire sull'altro ramo della pista.
        nominal = max(0.0, float(self.brush_radius) - clearance)
        free_right = self._free_distance(self.center[coarse], self.normal[coarse], self.brush_radius + 1.0)
        free_left = self._free_distance(self.center[coarse], -self.normal[coarse], self.brush_radius + 1.0)
        self.lateral_upper = periodic(np.clip(free_right - clearance, 0.0, nominal))
        self.lateral_lower = -periodic(np.clip(free_left - clearance, 0.0, nominal))

    def _free_distance(self, origins, directions, max_distance=None):
        """Distanza percorribile dal punto lungo la direzione, fino al primo pixel fuori pista."""
        max_distance = 3.0 * self.brush_radius if max_distance is None else max_distance
        steps = np.arange(0.0, max_distance, 1.0)
        xs = origins[:, 0:1] + directions[:, 0:1] * steps[None, :]
        ys = origins[:, 1:2] + directions[:, 1:2] * steps[None, :]
        blocked = ~self.drivable(xs, ys)
        return np.where(blocked.any(axis=1), steps[blocked.argmax(axis=1)], max_distance)

    def generate_checkpoints(self, count=NUM_CHECKPOINTS):
        """Checkpoint equidistanti lungo l'ascissa curvilinea; il checkpoint 0 è la linea di partenza."""
        spacing = self.length / count
        self.checkpoints = [tuple(self.point_at(i * spacing)) for i in range(count)]

    @property
    def checkpoint_spacing(self):
        return self.length / max(1, len(self.checkpoints))

    def index_at(self, arc_length):
        return int(round((arc_length % self.length) / self.step)) % len(self.center)

    def point_at(self, arc_length):
        return self.center[self.index_at(arc_length)]

    def project(self, x, y, hint=None):
        """Indice del punto di mezzeria più vicino. Con hint la ricerca è locale (piste che si incrociano)."""
        count = len(self.center)
        if hint is None:
            candidates = np.arange(count)
        else:
            candidates = (hint + np.arange(-PROJECTION_WINDOW_POINTS, PROJECTION_WINDOW_POINTS + 1)) % count
        delta = self.center[candidates] - (x, y)
        return int(candidates[np.argmin(np.einsum("ij,ij->i", delta, delta))])

    def lateral_offset(self, x, y, index):
        dx = x - self.center[index, 0]
        dy = y - self.center[index, 1]
        return float(dx * self.normal[index, 0] + dy * self.normal[index, 1])

    def drivable(self, xs, ys):
        """Versione vettoriale di get_pixel_safety."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        if self.mask is None:
            flat = [self.get_pixel_safety(px, py) for px, py in zip(xs.ravel(), ys.ravel())]
            return np.asarray(flat, dtype=bool).reshape(xs.shape)
        # +1 per il bordo; dopo il clip i valori sono >= 0 e la conversione a intero equivale a floor.
        ix = (np.clip(xs, -1.0, self.width) + 1.0).astype(np.intp)
        iy = (np.clip(ys, -1.0, self.height) + 1.0).astype(np.intp)
        return self._padded_mask[ix, iy]

    def get_pixel_safety(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            if self.mask is not None:
                return bool(self.mask[int(x), int(y)])
            color = self.surface.get_at((int(x), int(y)))
            return color[0] > 50
        return False
