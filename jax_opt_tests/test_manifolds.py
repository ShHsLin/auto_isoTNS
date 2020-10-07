from jax_opt import manifolds
import pytest
import jax
jax.config.update('jax_enable_x64', True)


class CheckManifolds():

    def __init__(self, m, descr, shape, tol):
        self.m = m  # example of a manifold
        self.descr = descr
        self.shape = shape  # shape of a tensor
        self.rng_key = jax.random.PRNGKey(0)
        self.u, self.rng_key = m.random(shape, self.rng_key, dtype=jax.numpy.complex128)  # point from a manifold
        self.v1, self.rng_key = m.random_tangent(self.u, self.rng_key)  # first tangent vector
        # import numpy
        # self.u = jax.numpy.array(numpy.load('/space/ga63zuh/Project/QGOpt/u.npy'))
        # self.v1 = jax.numpy.array(numpy.load('/space/ga63zuh/Project/QGOpt/v1.npy'))

        self.v2, self.rng_key = m.random_tangent(self.u, self.rng_key)  # second tangent vector
        self.zero = self.u * 0.  # zero vector
        self.tol = tol  # tolerance of a test

    def _proj_of_tangent(self):
        """
        Checking m.proj: Projection of a vector from the tangent space
        at some point.

        Args:

        Returns:
            error, float number
        """

        err = jax.numpy.linalg.norm(self.v1 - self.m.proj(self.u, self.v1),
                                    axis=(-2, -1)
                                   )
        return err

    def _inner_proj_matching(self):
        """
        Checking matching between m.inner and m.proj

        Args:

        Returns:
            error, float number
        """
        list_of_dtypes = [jax.numpy.complex64, jax.numpy.complex128]
        if self.u.dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")

        real_dtype = jax.numpy.float64 if self.u.dtype == jax.numpy.complex128 else jax.numpy.float32

        self.rng_key, *rng_subkeys = jax.random.split(self.rng_key, 3)
        xi_real = jax.random.normal(rng_subkeys[0], self.u.shape, dtype=real_dtype)
        xi_imag = jax.random.normal(rng_subkeys[1], self.u.shape, dtype=real_dtype)
        xi = xi_real + 1j * xi_imag

        xi_proj = self.m.proj(self.u, xi)
        first_inner = self.m.inner(self.u, xi_proj, self.v1)
        second_inner = self.m.inner(self.u, xi, self.v1)
        err = jax.numpy.abs(first_inner - second_inner)
        return err

    def _retraction(self):
        """
        Checking retraction
        Page 46, Nikolas Boumal, An introduction to optimization on smooth
        manifolds.
        1) Rx(0) = x (Identity mapping)
        2) DRx(o)[v] = v : introduce v->t*v, calculate err=dRx(tv)/dt|_{t=0}-v
        3) Presence of a new point in a manifold

        Args:

        Returns:
            list of errors, first two errors have float dtype, the last one
            has boolean dtype
        """

        dt = 1e-8  # dt for numerical derivative

        # transition along zero vector (first cond)
        err1 = self.u - self.m.retraction(self.u, self.zero)
        err1 = jax.numpy.linalg.norm(err1)

        # differential (second cond)
        retr = self.m.retraction(self.u, dt * self.v1)
        dretr = (retr - self.u) / dt
        err2 = jax.numpy.linalg.norm(dretr - self.v1)

        # presence of a new point in a manifold (third cond)
        err3 = self.m.is_in_manifold(self.m.retraction(self.u, self.v1),
                                                                tol=self.tol)
        return jax.numpy.float32(err1), jax.numpy.float32(err2), err3

    def _vector_transport(self):
        """
        Checking vector transport.
        Page 264, Nikolas Boumal, An introduction to optimization on smooth
        manifolds.
        1) transported vector lies in a new tangent space
        2) VT(x,0)[v] is the identity mapping on TxM.

        Args:

        Returns:
            list of errors, each error has float dtype
        """

        vt = self.m.vector_transport(self.u, self.v1, self.v2)
        err1 = vt - self.m.proj(self.m.retraction(self.u, self.v2), vt)
        err1 = jax.numpy.linalg.norm(err1, axis=(-2, -1))

        err2 = self.v1 - self.m.vector_transport(self.u, self.v1, self.zero)
        err2 = jax.numpy.linalg.norm(err2, axis=(-2, -1))
        # return tf.cast(err1, dtype=tf.float32), tf.cast(err2, dtype=tf.float32)
        return err1, err2

    def _egrad_to_rgrad(self):
        """
        Checking egrad_to_rgrad method.
        1) rgrad is in a tangent space
        2) <v1 egrad> = <v1 rgrad>_m (matching between egrad and rgrad)
        Args:

        Returns:
            list of errors, each error has float dtype
        """

        list_of_dtypes = [jax.numpy.complex64, jax.numpy.complex128]
        if self.u.dtype not in list_of_dtypes:
            raise ValueError("Incorrect dtype")

        real_dtype = jax.numpy.float64 if self.u.dtype == jax.numpy.complex128 else jax.numpy.float32

        self.rng_key, *rng_subkeys = jax.random.split(self.rng_key, 3)
        xi_real = jax.random.normal(rng_subkeys[0], self.u.shape, dtype=real_dtype)
        xi_imag = jax.random.normal(rng_subkeys[1], self.u.shape, dtype=real_dtype)
        xi = xi_real + 1j * xi_imag

        rgrad = self.m.egrad_to_rgrad(self.u, xi)
        err1 = rgrad - self.m.proj(self.u, rgrad)
        err1 = jax.numpy.linalg.norm(err1, axis=(-2, -1))

        err2 = jax.numpy.sum(jax.numpy.conj(self.v1) * xi, axis=(-2, -1)) -\
                self.m.inner(self.u, self.v1, rgrad)
        err2 = jax.numpy.abs(jax.numpy.real(err2))
        return err1, err2

    def checks(self):
        # TODO after checking: rewrite with asserts
        """
        Routine for pytest: checking tolerance of manifold functions
        """
        err = self._proj_of_tangent()
        assert err < self.tol, "Projection error for:{}.\
                    ".format(self.descr)

        if self.descr[1] not in ['log_cholesky']:
            err = self._inner_proj_matching()
            assert err < self.tol, "Inner/proj error for:{}.\
                    ".format(self.descr)

        err1, err2, err3 = self._retraction()
        assert err1 < self.tol, "Retraction (Rx(0) != x) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Retraction (DRx(o)[v] != v) error for:{}.\
                    ".format(self.descr)
        assert err3 == True, "Retraction (not in manifold) error for:{}.\
                    ".format(self.descr)

        err1, err2 = self._vector_transport()
        assert err1 < self.tol, "Vector transport (not in a TMx) error for:{}.\
                    ".format(self.descr)
        assert err2 < self.tol, "Vector transport (VT(x,0)[v] != v) error for:\
                    {}.".format(self.descr)

        err1, err2 = self._egrad_to_rgrad()
        if self.descr[0] not in ['ChoiMatrix', 'DensityMatrix']:
            assert err1 < self.tol, "Rgrad (not in a TMx) error for:{}.\
                    ".format(self.descr)
        if self.descr[1] not in ['log_cholesky']:
            assert err2 < self.tol, "Rgrad (<v1 egrad> != inner<v1 rgrad>) error \
                    for:{}.".format(self.descr)

#TODO find a problem with tests or/and PositiveCone manifold
testdata = [
    ('Stiefel', 'euclidean', manifolds.StiefelManifold(metric='euclidean'), (4, 4), 1.e-6),
    # ('ChoiMatrix', 'euclidean', manifolds.ChoiMatrix(metric='euclidean'), (4, 4), 1.e-6),
    # ('DensityMatrix', 'euclidean', manifolds.DensityMatrix(metric='euclidean'), (4, 4), 1.e-6),
    # ('HermitianMatrix', 'euclidean', manifolds.HermitianMatrix(metric='euclidean'), (4, 4), 1.e-6),
    #('PositiveCone', 'log_euclidean', manifolds.PositiveCone(metric='log_euclidean'), (4, 4), 1.e-5),
    #('PositiveCone', 'log_cholesky', manifolds.PositiveCone(metric='log_cholesky'), (4, 4), 1.e-5),
]

@pytest.mark.parametrize("name,metric,manifold,shape,tol", testdata)
def test_manifolds(name, metric, manifold, shape, tol):
    Test = CheckManifolds(manifold, (name, metric), shape, tol)
    Test.checks()
