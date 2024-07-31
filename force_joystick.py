# !/usr/bin/env python3
import casadi as cs
from horizon.transcriptions import integrators
import matplotlib.pyplot as plt
import numpy as np

class ForceJoystick:
    def __init__(self, dt, n_step=10, sys_dim=2, opt=None):

        self.__dim = sys_dim

        if opt is None:
            opt = dict()

        if 'mass' not in opt:
            opt['mass'] = np.ones(self.__dim)

        if 'spring' not in opt:
            opt['spring'] = np.zeros(self.__dim)

        if 'damp' not in opt:
            opt['damp'] = np.zeros(self.__dim)

        self.__mass = opt['mass']  # virtual mass
        self.__spring = opt['spring']  # damping coefficient
        self.__damp = opt['damp']  # spring coefficient

        self.__dt = dt

        self.__n_step = n_step

        self.__x_int = np.zeros([self.__dim * 2, self.__n_step])

        self.__position_ref = np.zeros([self.__dim])

        self.__init_integrator()

    def __parameteric_RK4(self, x, u, xdot, p):

        L = 0

        f_RK = cs.Function('f_RK', [x, u, p], [xdot, L])

        nx = x.size1()
        nv = u.size1()
        np = p.size1()

        X0_RK = cs.SX.sym('X0_RK', nx)
        U_RK = cs.SX.sym('U_RK', nv)
        DT_RK = cs.SX.sym('DT_RK', 1)
        P_RK = cs.SX.sym('P_RK', np)

        X_RK = X0_RK
        Q_RK = 0

        k1, k1_q = f_RK(X_RK, U_RK, P_RK)
        k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK, P_RK)
        k3, k3_q = f_RK(X_RK + DT_RK / 2. * k2, U_RK, P_RK)
        k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK, P_RK)

        X_RK = X_RK + DT_RK / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        Q_RK = Q_RK + DT_RK / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)

        f = cs.Function('F_RK', [X0_RK, U_RK, DT_RK, P_RK], [X_RK, Q_RK], ['x', 'u', 'dt', 'p'], ['f', 'qf'])
        return f

    def __init_integrator(self):

        # system definition
        p = cs.SX.sym('pos', self.__dim)
        v = cs.SX.sym('vel', self.__dim)
        F = cs.SX.sym('force', self.__dim)

        # initial position
        p0 = cs.SX.sym('pos_0', self.__dim)
        # v0 = cs.SX.sym('vel_0', self.__dim)
        v0 = np.zeros([self.__dim])

        # v0.assign([0, 0, 0])

        # mass-spring-damper
        x = cs.vertcat(p, v)

        xdot = cs.vertcat(v, (F - self.__damp * (v - v0) - self.__spring * (p - p0))/self.__mass)


        self.integrator = self.__parameteric_RK4(x, F, xdot, p0)

    def update(self, x_current, u_current):

        # integrate for n step
        for step_i in range(self.__n_step):
            # assuming that the input is constant (Force is constant for the whole integration)
            int_state = self.integrator(x_current, u_current, self.__dt, self.__position_ref)[0]

            x_current = int_state.full()

            self.__x_int[:, step_i] = x_current.flatten()

    def getIntegratedState(self):

        return self.__x_int

    def getDimension(self):

        return self.__dim

    def setPositionReference(self, pos_ref):

        self.__position_ref = pos_ref

if __name__ == '__main__':
    sys_dim = 3

    x0 = np.zeros([2 * sys_dim, 1]) #m
    F0 = np.zeros(sys_dim) #N/m
    x0[0] = 5

    dt = 0.01

    m_virtual = np.array([50, 50, 50])
    k_virtual = np.array([10, 10, 0])

    d_virtual = 2 * np.sqrt(k_virtual * m_virtual)

    n_steps = 10
    fj = ForceJoystick(dt,
                       n_step=n_steps,
                       sys_dim=sys_dim,
                       opt=dict(damp=d_virtual, spring=k_virtual))



    x_traj = np.zeros(sys_dim)
    v_traj = list()
    t_traj = list()
    t_curr = 0

    plt.ion()
    fig, ax = plt.subplots(1, 2)
    x, y, z = [],[], []


    future_sc_xy = list()
    future_sc_xz = list()
    alpha_vec = [1 - (i/n_steps) for i in range(n_steps)]

    for step_i in range(n_steps):
        future_sc_xy.append(ax[0].scatter(x, y, color='b', alpha = alpha_vec[step_i]))

    for step_i in range(n_steps):
        future_sc_xz.append(ax[1].scatter(x, z, color='b', alpha = alpha_vec[step_i]))

    sc_xy = ax[0].scatter(x,y, color='r')
    sc_xz = ax[1].scatter(x,z, color='r')

    plt.draw()
    lim = 10
    ax[0].set_title('xy')
    ax[0].set_xlim(-lim, lim)
    ax[0].set_ylim(-lim, lim)

    ax[1].set_title('xz')
    ax[1].set_xlim(-lim, lim)
    ax[1].set_ylim(-lim, lim)

    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)

    fj.setPositionReference(x0[:3, 0])
    x_curr = x0

    for t in range(10000):

        # if 80 < t < 100:
        #     F0 = np.array([50, 0, 50])
        # else:
        #     F0 = np.array([0, 0, 0])

        # if t > 80:
        #     fj.setPositionReference(np.array([0, 0, 0]))


        print("F applied: ", F0)

        fj.update(x_curr[:, 0], F0)
        x_curr = fj.getIntegratedState()

        t_curr = t_curr + dt


        # xy plane
        sc_xy.set_offsets(np.c_[x_curr[0, 0], x_curr[1, 0]])

        for step_i in range(1, n_steps):
            future_sc_xy[step_i].set_offsets(np.c_[x_curr[0, step_i],x_curr[1, step_i]])


        # xz plane
        sc_xz.set_offsets(np.c_[x_curr[1, 0], x_curr[2, 0]])

        for step_i in range(1, n_steps):
            future_sc_xz[step_i].set_offsets(np.c_[x_curr[1, step_i],x_curr[2, step_i]])

        fig.canvas.draw_idle()
        plt.pause(0.01)


    plt.show()

