import numpy as np
from b777_engine.b777_engine import get_b777_engine, plot_b777_engine
# from b777_engine import get_b777_engine, plot_b777_engine
from smt.surrogate_models import RMTB
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

xt, yt, dyt_dxt, xlimits = get_b777_engine()

interp = RMTB(
    num_ctrl_pts=15,
    xlimits=xlimits,
    nonlinear_maxiter=20,
    approx_order=2,
    # energy_weight=0e-14,
    energy_weight=1e-2,
    # regularization_weight=0e-18,
    regularization_weight=0,
    extrapolate=True,
    print_global=False,
    # print_global=True,
)
interp.set_training_values(xt, yt)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
interp.train()

# plot_b777_engine(xt, yt, xlimits, interp)



def b777_smt(m, h, t):
    m = np.clip(m, 0.0, 0.9)
    h = np.clip(h, 0.0, 15.0)
    t = np.clip(t, 0.0, 1.0)

    x = np.stack([m, h, t], axis=-1).reshape(1, 3)
    y = interp.predict_values(x)
    return np.asarray(y[0, 0]), np.asarray(y[0, 1])  # (tmax, sfc)


def b777_smt_derivs(m, h, t):
    m = np.clip(m, 0.0, 1.0)
    h = np.clip(h, 0.0, np.inf)
    t = np.clip(t, 0.0, 1.0)

    x = np.stack([m, h, t], axis=-1).reshape(1, 3)
    dy_dm = interp.predict_derivatives(x, 0)
    dy_dh = interp.predict_derivatives(x, 1)
    dy_dt = interp.predict_derivatives(x, 2)
    # each is (1, 2) → unpack derivatives
    return (
        dy_dm[0, 0], dy_dh[0, 0], dy_dt[0, 0],  # dtmax_dm, dtmax_dh, dtmax_dt
        dy_dm[0, 1], dy_dh[0, 1], dy_dt[0, 1],  # dsfc_dm, dsfc_dh, dsfc_dt
    )



def b777_fwd(m, h, t):
    tmax, sfc = b777_engine_model(m, h, t)
    return (tmax, sfc), (m, h, t, tmax, sfc)


def b777_bwd(residuals, g):
    m, h, t, tmax, sfc = residuals
    gtmax, gsfc = g

    out_types = tuple(
        jax.ShapeDtypeStruct((), jnp.float64) for _ in range(6)
    )
    derivs = jax.pure_callback(b777_smt_derivs, out_types, m, h, t)

    (dtmax_dm, dtmax_dh, dtmax_dt,
     dsfc_dm, dsfc_dh, dsfc_dt) = derivs

    # Chain rule
    dm = gtmax * dtmax_dm + gsfc * dsfc_dm
    dh = gtmax * dtmax_dh + gsfc * dsfc_dh
    dt = gtmax * dtmax_dt + gsfc * dsfc_dt

    return (dm, dh, dt)


@jax.custom_vjp
def b777_engine_model(m, h, t):
    out_types = (jax.ShapeDtypeStruct((), jnp.float64),
                 jax.ShapeDtypeStruct((), jnp.float64))
    tmax, sfc = jax.pure_callback(b777_smt, out_types, m, h, t)
    return tmax, sfc

b777_engine_model.defvjp(b777_fwd, b777_bwd)



if __name__ == "__main__":
    import time

    m = np.array(0.7)
    h = np.array(10.) # (km)
    t = np.array(0.5)

    # y_smt = b777_smt(m, h, t)
    t1 = time.time()
    y_jax = b777_engine_model(m, h, t)
    t2 = time.time()
    print("JAX prediction time: ", t2 - t1)


    # print("tmax-jax (N): ", y_jax[0])
    # print("sfc-jax (N/N/s): ", y_jax[1])
    # print("sfc-jax (kg/N/s): ", y_jax[1] * 0.101971621)

    t1 = time.time()

    dtmax_dm = jax.jacrev(lambda m: b777_engine_model(m, h, t)[0])(m)
    dsfc_dm = jax.jacrev(lambda m: b777_engine_model(m, h, t)[1])(m)
    # print("dtmax_dm: ", dtmax_dm)
    # print("dsfc_dm: ", dsfc_dm)

    dtmax_dh = jax.jacrev(lambda h: b777_engine_model(m, h, t)[0])(h)
    dsfc_dh = jax.jacrev(lambda h: b777_engine_model(m, h, t)[1])(h)
    # print("dtmax_dh: ", dtmax_dh)
    # print("dsfc_dh: ", dsfc_dh)

    dtmax_dt = jax.jacrev(lambda t: b777_engine_model(m, h, t)[0])(t)
    dsfc_dt = jax.jacrev(lambda t: b777_engine_model(m, h, t)[1])(t)
    # print("dtmax_dt: ", dtmax_dt)
    # print("dsfc_dt: ", dsfc_dt)

    t2 = time.time()
    print("JAX derivative time: ", t2 - t1)

    # derivs = b777_smt_derivs(m, h, t)

    # print("dtmax_dm: ", derivs["dtmax_dm"])
    # print("dtmax_dh: ", derivs["dtmax_dh"])
    # print("dtmax_dt: ", derivs["dtmax_dt"])
    # print("dsfc_dm: ", derivs["dsfc_dm"])
    # print("dsfc_dh: ", derivs["dsfc_dh"])
    # print("dsfc_dt: ", derivs["dsfc_dt"])

    # # check derivatives against finite difference
    # eps = 1e-6
    # m_eps = m + eps
    # h_eps = h + eps
    # t_eps = t + eps
    # tmax_fd_m = (b777_engine_model(m_eps, h, t)[0] - b777_engine_model(m, h, t)[0]) / eps
    # sfc_fd_m = (b777_engine_model(m_eps, h, t)[1] - b777_engine_model(m, h, t)[1]) / eps
    # tmax_fd_h = (b777_engine_model(m, h_eps, t)[0] - b777_engine_model(m, h, t)[0]) / eps
    # sfc_fd_h = (b777_engine_model(m, h_eps, t)[1] - b777_engine_model(m, h, t)[1]) / eps
    # tmax_fd_t = (b777_engine_model(m, h, t_eps)[0] - b777_engine_model(m, h, t)[0]) / eps
    # sfc_fd_t = (b777_engine_model(m, h, t_eps)[1] - b777_engine_model(m, h, t)[1]) / eps

    # print("dtmax_dm (fd): ", tmax_fd_m)
    # print("dsfc_dm (fd): ", sfc_fd_m)
    # print("dtmax_dh (fd): ", tmax_fd_h)
    # print("dsfc_dh (fd): ", sfc_fd_h)
    # print("dtmax_dt (fd): ", tmax_fd_t)
    # print("dsfc_dt (fd): ", sfc_fd_t)