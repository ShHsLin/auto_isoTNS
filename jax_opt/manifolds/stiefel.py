from jax_opt.manifolds import base_manifold
import jax
from jax_opt.manifolds.utils import adj
jax.config.update('jax_enable_x64', True)


class StiefelManifold(base_manifold.Manifold):
    """The complex Stiefel manifold (St(n, p) is the manifold of complex
    valued isometric matrices of size n x p). One can use it to perform
    moving of points and vectors along the manifold.

    The geometry of the complex Stiefel manifold is taken from

    Sato, H., & Iwai, T. (2013, December). A complex singular value
    decomposition algorithm based on the Riemannian Newton method.
    In 52nd IEEE Conference on Decision and Control (pp. 2972-2978). IEEE.

    Another paper, which was used as a guide is

    Edelman, A., Arias, T. A., & Smith, S. T. (1998). The geometry of
    algorithms with orthogonality constraints. SIAM journal on Matrix
    Analysis and Applications, 20(2), 303-353.

    Args:
        retraction: string specifies type of retraction. Defaults to
            'svd'. Types of retraction are available: 'svd', 'cayley', 'qr'.
        metric: string specifies type of metric, Defaults to 'euclidean'.
            Types of metrics are available: 'euclidean', 'canonical'.

    Notes:
        All methods of this class operates with tensors of shape (..., n, p),
        where (...) enumerates manifold (can be any shaped), (n, p)
        is the shape of a particular matrix (e.g. an element of the complex
        Stiefel manifold or its tangent vector)."""

    def __init__(self, retraction='svd',
                 metric='euclidean'):

        self.rank = 2
        self.quotient = False
        list_of_metrics = ['euclidean', 'canonical']
        list_of_retractions = ['svd', 'cayley', 'qr']

        if metric not in list_of_metrics:
            raise ValueError("Incorrect metric")
        if retraction not in list_of_retractions:
            raise ValueError("Incorrect retraction")

        super(StiefelManifold, self).__init__(retraction, metric)

    def inner(self, u, vec1, vec2):
        """Returns manifold wise inner product of vectors from
        a tangent space.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold.
            vec1: complex valued tensor of shape (..., n, p),
                a set of tangent vectors from the complex
                Stiefel manifold.
            vec2: complex valued tensor of shape (..., n, p),
                a set of tangent vectors from the complex
                Stiefel manifold.

        Returns:
            complex valued tensor of shape (..., 1, 1),
            manifold wise inner product"""

        if self._metric == 'euclidean':
            s_sq = jax.numpy.trace(adj(vec1) @ vec2, axis1=-1, axis2=-2)[...,
                                                                         None,
                                                                         None]
        elif self._metric == 'canonical':
            G = jax.numpy.eye(u.shape[-2], dtype=u.dtype) - u @ adj(u) / 2
            s_sq = jax.numpy.trace(adj(vec1) @ G @ vec2, axis1=-1, axis2=-2)[...,
                                                                             None,
                                                                             None]
        return jax.numpy.real(s_sq)

    def proj(self, u, vec):
        """Returns projection of vectors on a tangent space
        of the complex Stiefel manifold.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold.
            vec: complex valued tensor of shape (..., n, p),
                a set of vectors to be projected.

        Returns:
            complex valued tensor of shape (..., n, p),
            a set of projected vectors"""

        return vec - 0.5 * u @ (adj(u) @ vec + adj(vec) @ u)

    def egrad_to_rgrad(self, u, egrad):
        """Returns the Riemannian gradient from an Euclidean gradient.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold.
            egrad: complex valued tensor of shape (..., n, p),
                a set of Euclidean gradients.

        Returns:
            complex valued tensor of shape (..., n, p),
            the set of Reimannian gradients."""

        if self._metric == 'euclidean':
            return egrad - 0.5 * u @ (adj(u) @ egrad + adj(egrad) @ u)

        elif self._metric == 'canonical':
            return egrad - u @ adj(egrad) @ u

    def retraction(self, u, vec):
        """Transports a set of points from the complex Stiefel
        manifold via a retraction map.

        Args:
            u: complex valued tensor of shape (..., n, p), a set
                of points to be transported.
            vec: complex valued tensor of shape (..., n, p),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, p),
            a set of transported points."""

        if self._retraction == 'svd':
            new_u = u + vec
            # _, v, w = tf.linalg.svd(new_u)
            v, _, wh = jax.numpy.linalg.svd(new_u, full_matrices=False)
            return v @ wh

        elif self._retraction == 'cayley':
            W = vec @ adj(u) - 0.5 * u @ (adj(u) @ vec @ adj(u))
            W = W - adj(W)
            Id = jax.numpy.eye(W.shape[-1], dtype=W.dtype)
            return jax.numpy.linalg.inv(Id - W / 2) @ (Id + W / 2) @ u

        elif self._retraction == 'qr':
            new_u = u + vec
            q, r = jax.numpy.linalg.qr(new_u)
            diag = jax.numpy.diag(r)
            sign = jax.numpy.sign(diag)[..., None, :]
            return q * sign

    def vector_transport(self, u, vec1, vec2):
        """Returns a vector tranported along an another vector
        via vector transport.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold, starting points.
            vec1: complex valued tensor of shape (..., n, p),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, p),
                a set of direction vectors.

        Returns:
            complex valued tensor of shape (..., n, p),
            a set of transported vectors."""

        new_u = self.retraction(u, vec2)
        return self.proj(new_u, vec1)

    def retraction_transport(self, u, vec1, vec2):
        """Performs a retraction and a vector transport simultaneously.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a set of points from the complex Stiefel
                manifold, starting points.
            vec1: complex valued tensor of shape (..., n, p),
                a set of vectors to be transported.
            vec2: complex valued tensor of shape (..., n, p),
                a set of direction vectors.

        Returns:
            two complex valued tensors of shape (..., n, p),
            a set of transported points and vectors."""

        new_u = self.retraction(u, vec2)
        return new_u, self.proj(new_u, vec1)

    def random(self, shape, rng_key, dtype=jax.numpy.complex64):
        """Returns a set of points from the complex Stiefel
        manifold generated randomly.

        Args:
            shape: tuple of integer numbers (..., n, p),
                shape of a generated matrix.
            rng_key: the jax random number generator
            dtype: type of an output tensor, can be
                either jax.numpy.complex64 or jax.numpy.complex128.

        Returns:
            complex valued tensor of shape (..., n, p),
            a generated matrix."""

        list_of_dtypes = [jax.numpy.complex64, jax.numpy.complex128]

        if dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")

        real_dtype = jax.numpy.float64 if dtype == jax.numpy.complex128 else jax.numpy.float32

        rng_key, *rng_subkeys = jax.random.split(rng_key, 3)
        u_real = jax.random.normal(rng_subkeys[0], shape, dtype=real_dtype)
        u_imag = jax.random.normal(rng_subkeys[1], shape, dtype=real_dtype)
        u = u_real + 1j * u_imag
        u, _ = jax.numpy.linalg.qr(u)
        return u, rng_key

    def random_tangent(self, u, rng_key):
        """Returns a set of random tangent vectors to points from
        the complex Stiefel manifold.

        Args:
            u: complex valued tensor of shape (..., n, p), points
                from the complex Stiefel manifold.
            rng_key: the jax random number generator

        Returns:
            complex valued tensor, set of tangent vectors to u."""

        list_of_dtypes = [jax.numpy.complex64, jax.numpy.complex128]

        if u.dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")

        real_dtype = jax.numpy.float64 if u.dtype == jax.numpy.complex128 else jax.numpy.float32

        rng_key, *rng_subkeys = jax.random.split(rng_key, 3)
        vec_real = jax.random.normal(rng_subkeys[0], u.shape, dtype=real_dtype)
        vec_imag = jax.random.normal(rng_subkeys[1], u.shape, dtype=real_dtype)
        vec = vec_real + 1j * vec_imag

        vec = self.proj(u, vec)
        return vec, rng_key

    def is_in_manifold(self, u, tol=1e-5):
        """Checks if a point is in the Stiefel manifold or not.

        Args:
            u: complex valued tensor of shape (..., n, p),
                a point to be checked.
            tol: small real value showing tolerance.

        Returns:
            bolean tensor of shape (...)."""

        Id = jax.numpy.eye(u.shape[-1], dtype=u.dtype)
        udagu = adj(u) @ u
        diff = Id - udagu
        diff_norm = jax.numpy.linalg.norm(diff, axis=(-1,-2))
        udagu_norm = jax.numpy.linalg.norm(udagu, axis=(-1,-2))
        Id_norm = jax.numpy.linalg.norm(Id, axis=(-1,-2))
        rel_diff = jax.numpy.abs(diff_norm / jax.numpy.sqrt(Id_norm * udagu_norm))
        return tol > rel_diff
