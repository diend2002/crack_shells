# src/crack_geometry.py
import torch

class CrackGeometry:
    def __init__(self, crack_type='center', crack_param=None, smoothing=1e-2):
        self.crack_type = crack_type
        self.crack_param = {} if crack_param is None else dict(crack_param)
        self.smoothing = float(smoothing)
        self.A, self.B = self._endpoints_from_params(self.crack_type, self.crack_param)
        # Do NOT precompute t_hat/n_hat/L on a fixed device here

    def _endpoints_from_params(self, crack_type, p):
        L = float(p.get('length', 1.0))
        angle = float(p.get('angle', 0.0))
        cx, cy = p.get('center', (0.0, 0.0)) if crack_type != 'edge' else p.get('start', (0.0, 0.0))
        # build on default dtype, no device
        c = torch.tensor([cx, cy], dtype=torch.get_default_dtype())
        t = torch.stack((torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle)))).to(c.dtype)
        if crack_type == 'edge':
            A = c
            B = c + L * t
        else:  # 'center' or 'inclined'
            half = 0.5 * L
            A = c - half * t
            B = c + half * t
        return A, B

    def _signed_distance_to_segment(self, P):
        # Move endpoints to P’s device/dtype each call
        A = self.A.to(P.device, P.dtype)
        B = self.B.to(P.device, P.dtype)

        AB = B - A
        L = torch.linalg.norm(AB)
        t_hat = AB / (L + 1e-16)
        n_hat = torch.stack((-t_hat[1], t_hat[0]))  # left-hand normal, same device/dtype

        AP = P - A
        s = AP @ t_hat
        d = AP @ n_hat

        s_lt0 = s < 0
        s_gtL = s > L
        s_clamp = torch.where(s_lt0, -s, torch.where(s_gtL, s - L, torch.zeros_like(s)))

        eps_dist = 1e-12
        dist_seg = torch.sqrt(d*d + s_clamp*s_clamp + eps_dist)

        eps_sign_inner = max(self.smoothing, 1e-3)
        sgn_smooth = torch.tanh(d / eps_sign_inner)
        phi = sgn_smooth * dist_seg
        return phi

    def sdf(self, points):
        return self._signed_distance_to_segment(points)

    def strong_embedding(self, points, gate_eps=1e-3):
        phi = self.sdf(points)
        r1 = torch.linalg.norm(points - self.A.to(points.device, points.dtype), dim=1)
        r2 = torch.linalg.norm(points - self.B.to(points.device, points.dtype), dim=1)
        Fgate = (r1 * r2) / (r1 * r2 + gate_eps)
        return Fgate * torch.tanh(phi / max(self.smoothing, 1e-3))
    
    def endpoints(self, like: torch.Tensor | None = None):
        """Return (A,B). If `like` is given, cast to its dtype/device."""
        if like is None:
            return self.A, self.B
        return self.A.to(like.device, like.dtype), self.B.to(like.device, like.dtype)
