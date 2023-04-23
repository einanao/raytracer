from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt


def unit_norm(v):
    return v / np.linalg.norm(v)


def sample_in_unit_sphere(d=3):
    while True:
        p = np.random.random(d) * 2 - 1
        if p.dot(p) < 1:
            return p


def sample_on_unit_sphere():
    return unit_norm(sample_in_unit_sphere())


def reflect(v, n):
    return v - 2 * n * v.dot(n)


def refract(v, n, index_ratio):
    cos_theta = -v.dot(n)
    cos_theta = min(cos_theta, 1)
    r_out_perp = index_ratio * (v + cos_theta * n)
    r_out_parallel = -np.sqrt(np.abs(1 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


def cross(u, v):
    return np.array(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )


class Ray(object):
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self._unit_norm_direction = None

    def unit_norm_direction(self):
        if self._unit_norm_direction is None:
            self._unit_norm_direction = unit_norm(self.direction)
        return self._unit_norm_direction

    def at(self, t):
        return self.origin + t * self.direction


class Image(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        self.xs, self.ys = [a.T for a in np.meshgrid(x, y)]
        self.reset()

    def reset(self):
        self.data = np.zeros((self.width, self.height, 3))
        self.n_samples = np.zeros((self.width, self.height))

    def display(self, gamma=2):
        fig, ax = plt.subplots()
        ax.axis("off")
        img = self.data / self.n_samples[:, :, np.newaxis]
        img = np.power(img, 1 / gamma)
        img = (img * 255).astype(np.uint8)
        img = img[:, ::-1, :]
        img = np.swapaxes(img, 0, 1)
        ax.imshow(img[:, :, :])
        plt.show()

    def raytrace(self, ray_color, world, cam, n_samples=1):
        for x in range(self.width):
            for y in range(self.height):
                u = x / self.width
                v = y / self.height
                for _ in range(n_samples):
                    noisy_u = u + np.random.random() / self.width
                    noisy_v = v + np.random.random() / self.height
                    ray = cam.get_ray(noisy_u, noisy_v)
                    color = ray_color(ray, world)
                    self.data[x, y, :] += color
        self.n_samples += n_samples


class Hittable(object):
    @abstractmethod
    def hit(ray, t_min, t_max):
        raise NotImplementedError


class HittableList(object):
    def __init__(self, objects):
        self.objects = objects

    def hit(self, ray, t_min, t_max):
        closest_rec = None
        for obj in self.objects:
            rec = obj.hit(ray, t_min, t_max)
            if rec is not None and (closest_rec is None or rec.t < closest_rec.t):
                closest_rec = rec
        return closest_rec


class HitRecord(object):
    def __init__(self, p, normal, t, material):
        self.p = p
        self.normal = normal
        self.t = t
        self.material = material
        self.front_face = None

    def set_face_normal(self, ray, outward_normal):
        self.front_face = ray.direction.dot(outward_normal) < 0
        self.normal = outward_normal * self.front_face.astype(float)


class Material(object):
    def __init__(self, albedo):
        self.attenuation = albedo

    @abstractmethod
    def _scatter(self, r_in, rec):
        raise NotImplementedError

    def scatter(self, r_in, rec):
        scatter_direction = self._scatter(r_in, rec)
        if scatter_direction is None:
            return None
        scattered = Ray(rec.p, scatter_direction)
        return scattered, self.attenuation


class Lambertian(Material):
    def _scatter(self, r_in, rec):
        scatter_direction = rec.normal + sample_on_unit_sphere()
        if np.allclose(scatter_direction, 0):
            scatter_direction = rec.normal
        return scatter_direction


class Metal(Material):
    def __init__(self, albedo, fuzz):
        super().__init__(albedo)
        self.fuzz = fuzz

    def _scatter(self, r_in, rec):
        v = reflect(r_in.direction, rec.normal)
        v += sample_in_unit_sphere() * self.fuzz
        return v if v.dot(rec.normal) > 0 else None


class Dielectric(Material):
    def __init__(self, index):
        super().__init__(albedo=np.ones(3))
        self.index = index

    def _scatter(self, r_in, rec):
        index_ratio = 1 / self.index if rec.front_face else self.index
        unit_in = unit_norm(r_in.direction)
        cos_theta = -unit_in.dot(rec.normal)
        cos_theta = min(cos_theta, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        reflected = reflect(unit_in, rec.normal)
        r = self.reflectance(cos_theta)
        if index_ratio * sin_theta > 1 or np.random.random() < r:
            return reflected
        else:
            refracted = refract(unit_in, rec.normal, index_ratio)
            return refracted

    def reflectance(self, cos_theta):
        r0 = (1 - self.index) / (1 + self.index)
        r0 = r0**2
        return r0 + (1 - r0) * (1 - cos_theta) ** 5


class Sphere(Hittable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        h = ray.direction.dot(oc)
        c = oc.dot(oc) - self.radius**2
        discriminant = h**2 - a * c
        if discriminant < 0:
            return None
        sqrtd = np.sqrt(discriminant)
        root = (-h - sqrtd) / a
        if root < t_min or root > t_max:
            root = (-h + sqrtd) / a
            if root < t_min or root > t_max:
                return None
        p = ray.at(root)
        outward_normal = (p - self.center) / self.radius
        rec = HitRecord(p, outward_normal, root, self.material)
        rec.set_face_normal(ray, outward_normal)
        return rec


class Camera(object):
    def __init__(
        self,
        lookfrom,
        lookat,
        vup,
        aspect_ratio,
        vfov,
        aperture,
        focus_dist,
    ):
        w = unit_norm(lookfrom - lookat)
        self.u = unit_norm(cross(vup, w))
        self.v = cross(w, self.u)
        viewport_height = np.tan(vfov / 2) * 2
        viewport_width = viewport_height * aspect_ratio
        self.origin = lookfrom
        self.horizontal = viewport_width * self.u * focus_dist
        self.vertical = viewport_height * self.v * focus_dist
        self.lower_left_corner = (
            self.origin - self.horizontal / 2 - self.vertical / 2 - focus_dist * w
        )
        self.lens_radius = aperture / 2

    def get_ray(self, u, v):
        offset = sample_in_unit_sphere(d=2) * self.lens_radius
        offset = self.u * offset[0] + self.v * offset[1]
        origin = self.origin + offset
        direction = (
            self.lower_left_corner + u * self.horizontal + v * self.vertical - origin
        )
        return Ray(origin, direction)
