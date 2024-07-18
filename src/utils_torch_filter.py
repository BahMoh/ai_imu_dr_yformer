import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from termcolor import cprint
from utils_numpy_filter import NUMPYIEKF
from utils import prepare_data

# from exp.exp_informer import Exp_Informer
from models.model import Informer, Yformer, Yformer_skipless

class InitProcessCovNet(torch.nn.Module):

        def __init__(self):
            super(InitProcessCovNet, self).__init__()

            self.beta_process = 3*torch.ones(2).double()
            self.beta_initialization = 3*torch.ones(2).double()

            self.factor_initial_covariance = torch.nn.Linear(1, 6, bias=False).double()
            """parameters for initializing covariance"""
            self.factor_initial_covariance.weight.data[:] /= 10

            self.factor_process_covariance = torch.nn.Linear(1, 6, bias=False).double()
            """parameters for process noise covariance"""
            self.factor_process_covariance.weight.data[:] /= 10
            self.tanh = torch.nn.Tanh()

        def forward(self, iekf):
            return

        def init_cov(self, iekf):
            alpha = self.factor_initial_covariance(torch.ones(1).double()).squeeze()
            beta = 10**(self.tanh(alpha))
            return beta

        def init_processcov(self, iekf):
            alpha = self.factor_process_covariance(torch.ones(1).double())
            beta = 10**(self.tanh(alpha))
            return beta


class MesNet(torch.nn.Module):
        def __init__(self):
            super(MesNet, self).__init__()
            self.beta_measurement = 3*torch.ones(2).float()
            self.tanh = torch.nn.Tanh()
            # self.cov_net = torch.nn.Sequential(
            #            torch.nn.Conv1d(6, 32, 5),
            #            torch.nn.ReplicationPad1d(4),
            #            torch.nn.ReLU(),
            #            torch.nn.Dropout(p=0.5),
            #            torch.nn.Conv1d(32, 32, 5, dilation=3),
            #            torch.nn.ReplicationPad1d(4),
            #            torch.nn.ReLU(),
            #            torch.nn.Dropout(p=0.5),
            #            ).double()
            
            self.cov_net = Yformer(
                enc_in=6,
                dec_in=6,
                c_out=32,
                seq_len=6000,
                label_len=48,
                out_len=0, # pred_len
                factor=3,
                d_model=64,
                n_heads=3,
                e_layers=2,
                d_layers=2,
                d_ff=64,
                dropout=0.05,
                attn='prob',
                embed='learned',
                freq='h',
                activation='gelu',
                output_attention=False,
                distil=True,
                device=torch.device('cuda:0'if torch.cuda.is_available() else "cpu"),
                ).float()
            # self.cov_net = Yformer(
            #     enc_in=6,
            #     dec_in=6,
            #     c_out=32,
            #     seq_len=6000,
            #     label_len=48,
            #     out_len=0, # pred_len
            #     factor=3,
            #     d_model=128,
            #     n_heads=3,
            #     e_layers=2,
            #     d_layers=2,
            #     d_ff=64,
            #     dropout=0.05,
            #     attn='prob',
            #     embed='learned',
            #     freq='h',
            #     activation='gelu',
            #     output_attention=False,
            #     distil=True,
            #     device=torch.device('cuda:0'if torch.cuda.is_available() else "cpu"),
            #     ).float()
            "CNN for measurement covariance"
            self.cov_lin = torch.nn.Sequential(torch.nn.Linear(32, 2),
                                              torch.nn.Tanh(),
                                              ).float()
            self.cov_lin[0].bias.data[:] /= 100
            self.cov_lin[0].weight.data[:] /= 100

        def forward(self, u, iekf):
            # print(u.shape, "u.shape")
            # y_cov = self.cov_net(u, u, enc_self_mask=True, dec_self_mask=True, dec_enc_mask=True).transpose(0, 2).squeeze()
            u = u.float()
            y_cov = self.cov_net(u, u).transpose(0, 2).squeeze()
            y_cov = y_cov.transpose(-1, -2)
            # print(y_cov.shape)
            z_cov = self.cov_lin(y_cov)
            z_cov_net = self.beta_measurement.unsqueeze(0)*z_cov
            measurements_covs = (iekf.cov0_measurement.unsqueeze(0) * (10**z_cov_net))
            # print(f"u {u.shape}")                                    # u torch.Size([1, 6, 6000])
            # print(f"y_cov {y_cov.shape}")                            # y_cov torch.Size([6000, 32])
            # print(f"self.cov_net(u) {self.cov_net(u).shape}")        # self.cov_net(u) torch.Size([1, 32, 6000])
            # print(f"self.beta_measurement {self.beta_measurement}")  # self.beta_measurement tensor([3., 3.], dtype=torch.float64)
            # print(f"z_cov {z_cov.shape}")                            # z_cov torch.Size([6000, 2])
            # print(f"z_cov_net {z_cov_net.shape}")                    # z_cov_net torch.Size([6000, 2])
            # print(f"measurements_covs {measurements_covs.shape}")    # measurements_covs torch.Size([6000, 2])
            # input("Enter Sth")                                       # Enter Sth
            return measurements_covs


class TORCHIEKF(torch.nn.Module, NUMPYIEKF):
    # Id1 = torch.eye(1).double()
    # Id2 = torch.eye(2).double()
    # Id3 = torch.eye(3).double()
    # Id6 = torch.eye(6).double()
    # IdP = torch.eye(21).double()
    Id1 = torch.eye(1).float()
    Id2 = torch.eye(2).float()
    Id3 = torch.eye(3).float()
    Id6 = torch.eye(6).float()
    IdP = torch.eye(21).float()


    def __init__(self, parameter_class=None):
        torch.nn.Module.__init__(self)
        NUMPYIEKF.__init__(self, parameter_class=None)

        # mean and standard deviation of parameters for normalizing inputs
        self.u_loc = None
        self.u_std = None
        self.initprocesscov_net = InitProcessCovNet()
        self.mes_net = MesNet()
        self.cov0_measurement = None

        # modified parameters
        # self.IdP = torch.eye(self.P_dim).double()
        self.IdP = torch.eye(self.P_dim).float()

        if parameter_class is not None:
            self.filter_parameters = parameter_class()
            self.set_param_attr()

    def set_param_attr(self):
        # get a list of attribute only
        attr_list = [a for a in dir(self.filter_parameters) if not a.startswith('__')
                     and not callable(getattr(self.filter_parameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(self.filter_parameters, attr))

        self.Q = torch.diag(torch.Tensor([self.cov_omega, self.cov_omega, self. cov_omega,
                                           self.cov_acc, self.cov_acc, self.cov_acc,
                                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])
                            ).float()
                            # ).double()
        # self.cov0_measurement = torch.Tensor([self.cov_lat, self.cov_up]).double()
        self.cov0_measurement = torch.Tensor([self.cov_lat, self.cov_up]).float()
    def run(self, t, u,  measurements_covs, v_mes, p_mes, N, ang0):

        dt = t[1:] - t[:-1]  # (s)
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P = self.init_run(dt, u, p_mes, v_mes,
                                       N, ang0)

        for i in range(1, N):
            Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i = \
                self.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1], Rot_c_i[i-1],
                               t_c_i[i-1], P, u[i], dt[i-1])

            Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
                self.update(Rot_i, v_i, p_i, b_omega_i, b_acc_i, Rot_c_i_i, t_c_i_i, P_i,
                            u[i], i, measurements_covs[i])
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def init_run(self, dt, u, p_mes, v_mes, N, ang0):
            Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = \
                self.init_saved_state(dt, N, ang0)
            Rot[0] = self.from_rpy(ang0[0], ang0[1], ang0[2])
            v[0] = v_mes[0]
            P = self.init_covariance()
            return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def init_covariance(self):
        beta = self.initprocesscov_net.init_cov(self)
        P = torch.zeros(self.P_dim, self.P_dim).double()
        P[:2, :2] = self.cov_Rot0*beta[0]*self.Id2  # no yaw error
        P[3:5, 3:5] = self.cov_v0*beta[1]*self.Id2
        P[9:12, 9:12] = self.cov_b_omega0*beta[2]*self.Id3
        P[12:15, 12:15] = self.cov_b_acc0*beta[3]*self.Id3
        P[15:18, 15:18] = self.cov_Rot_c_i0*beta[4]*self.Id3
        P[18:21, 18:21] = self.cov_t_c_i0*beta[5]*self.Id3
        return P


    def init_saved_state(self, dt, N, ang0):
        Rot = dt.new_zeros(N, 3, 3)
        v = dt.new_zeros(N, 3)
        p = dt.new_zeros(N, 3)
        b_omega = dt.new_zeros(N, 3)
        b_acc = dt.new_zeros(N, 3)
        Rot_c_i = dt.new_zeros(N, 3, 3)
        t_c_i = dt.new_zeros(N, 3)
        # Rot_c_i[0] = torch.eye(3).double()
        Rot_c_i[0] = torch.eye(3).float()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i

    def propagate(self, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, Rot_c_i_prev, t_c_i_prev,
                  P_prev, u, dt):
        Rot_prev = Rot_prev.clone().float()
        acc_b = (u[3:6] - b_acc_prev).float()
        acc = (Rot_prev.mv(acc_b) + self.g).float()
        v = (v_prev + acc * dt).float()
        p = (p_prev + v_prev.clone() * dt + 1/2 * acc * dt**2).float()

        omega = (u[:3] - b_omega_prev)*dt.float()
        Rot = Rot_prev.mm(self.so3exp(omega)).float()

        b_omega = b_omega_prev.float()
        b_acc = b_acc_prev.float()
        Rot_c_i = Rot_c_i_prev.clone().float()
        t_c_i = t_c_i_prev.float()

        P = self.propagate_cov(P_prev, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev,
                               u, dt).float()
        return Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P

    def propagate_cov(self, P, Rot_prev, v_prev, p_prev, b_omega_prev, b_acc_prev, u,
                      dt):

        F = P.new_zeros(self.P_dim, self.P_dim).float()
        G = P.new_zeros(self.P_dim, self.Q.shape[0]).float()
        Q = self.Q.clone().float()
        F[3:6, :3] = self.skew(self.g).float()
        F[6:9, 3:6] = self.Id3.float()
        G[3:6, 3:6] = Rot_prev.float()
        F[3:6, 12:15] = -Rot_prev.float()
        v_skew_rot = self.skew(v_prev).mm(Rot_prev).float()
        p_skew_rot = self.skew(p_prev).mm(Rot_prev).float()
        G[:3, :3] = Rot_prev.float()
        G[3:6, :3] = v_skew_rot.float()
        G[6:9, :3] = p_skew_rot.float()
        F[:3, 9:12] = -Rot_prev.float()
        F[3:6, 9:12] = -v_skew_rot.float()
        F[6:9, 9:12] = -p_skew_rot.float()
        G[9:12, 6:9] = self.Id3.float()
        G[12:15, 9:12] = self.Id3.float()
        G[15:18, 12:15] = self.Id3.float()
        G[18:21, 15:18] = self.Id3.float()

        F = F * dt.float()
        G = G * dt.float()
        F_square = F.mm(F).float()
        F_cube = F_square.mm(F).float()
        Phi = (self.IdP + F + 1/2*F_square + 1/6*F_cube).float()
        P_new = Phi.mm(P.float() + G.mm(Q).mm(G.t())).mm(Phi.t()).float()
        return P_new.float()

    def update(self, Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, u, i, measurement_cov):
        # orientation of body frame
        Rot_body = Rot.mm(Rot_c_i).float()
        # velocity in imu frame
        v_imu = Rot.t().mv(v).float()
        omega = (u[:3] - b_omega).float()
        # velocity in body frame
        v_body = (Rot_c_i.t().mv(v_imu) + self.skew(t_c_i).mv(omega)).float()
        Omega = self.skew(omega).float()
        # Jacobian in car frame
        H_v_imu = Rot_c_i.t().mm(self.skew(v_imu)).float()
        H_t_c_i = self.skew(t_c_i).float()

        H = P.new_zeros(2, self.P_dim).float()
        H[:, 3:6] = Rot_body.t()[1:].float()
        H[:, 15:18] = H_v_imu[1:].float()
        H[:, 9:12] = H_t_c_i[1:].float()
        H[:, 18:21] = -Omega[1:].float()
        r = - v_body[1:].float()
        R = torch.diag(measurement_cov).float()

        Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up = \
            self.state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R)
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up


    @staticmethod
    def state_and_cov_update(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P, H, r, R):
        S = (H.mm(P).mm(H.t()) + R).float()
        
        # print(f"S.shape {S.shape}")                       # S.shape torch.Size([2, 2])
        # print(f"P.mm(H.t()).t() {P.mm(H.t()).t().shape}") # P.mm(H.t()).t() torch.Size([2, 21])
        # print(f"P.mm(H.t()) {P.mm(H.t()).shape}")         # P.mm(H.t()) torch.Size([21, 2])
        # print(f"H {H.shape}")                             # H torch.Size([2, 21])
        # print(f"P {P.shape}")                             # P torch.Size([21, 21])
        # print(f"r {r.shape}")                             # r torch.Size([2])
        # print(f"R {R.shape}")                             # R torch.Size([2, 2])

        Kt = torch.torch.linalg.solve(S.double(), P.double().mm(H.double().t()).t()).double()
        Kt = Kt.float()
        # Kt = torch.linalg.solve(S, H.mm(P).t()).t()
        K = Kt.t().float()
        dx = K.mv(r.view(-1)).float()

        dR, dxi = TORCHIEKF.sen3exp(dx[:9])
        dv = dxi[:, 0].float()
        dp = dxi[:, 1].float()
        Rot_up = dR.mm(Rot).float()
        v_up = (dR.mv(v) + dv).float()
        p_up = (dR.mv(p) + dp).float()


        b_omega_up = (b_omega + dx[9:12]).float()
        b_acc_up = (b_acc + dx[12:15]).float()

        dR = TORCHIEKF.so3exp(dx[15:18]).float()
        Rot_c_i_up = dR.mm(Rot_c_i).float()
        t_c_i_up = (t_c_i + dx[18:21]).float()

        I_KH = (TORCHIEKF.IdP - K.mm(H)).float()
        P_upprev = (I_KH.mm(P).mm(I_KH.t()) + K.mm(R).mm(K.t())).float()
        P_up = (P_upprev + P_upprev.t())/2
        return Rot_up, v_up, p_up, b_omega_up, b_acc_up, Rot_c_i_up, t_c_i_up, P_up

    @staticmethod
    def skew(x):
        X = torch.Tensor([[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]]).float()
        return X

    @staticmethod
    def rot_from_2_vectors(v1, v2):
        """ Returns a Rotation matrix between vectors 'v1' and 'v2'    """
        v1 = v1/torch.norm(v1).float()
        v2 = v2/torch.norm(v2).float()
        v = torch.cross(v1, v2).float()
        cosang = v1.matmul(v2).float()
        sinang = torch.norm(v).float()
        Rot = (TORCHIEKF.Id3 + TORCHIEKF.skew(v) + \
              TORCHIEKF.skew(v).mm(TORCHIEKF.skew(v))*(1-cosang)/(sinang**2)).float()
        return Rot

    @staticmethod
    def sen3exp(xi):
        phi = xi[:3].float()
        angle = torch.norm(phi).float()

        # Near |phi|==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).float()
            J = (TORCHIEKF.Id3 + 0.5 * skew_phi).float()
            Rot = (TORCHIEKF.Id3 + skew_phi).float()
        else:
            axis = (phi / angle).float()
            skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]]).float()
            s = torch.sin(angle).float()
            c = torch.cos(angle).float()

            J = ((s / angle) * TORCHIEKF.Id3 + (1 - s / angle) * TORCHIEKF.outer(axis, axis)\
                   + ((1 - c) / angle) * skew_axis).float()
            Rot = (c * TORCHIEKF.Id3 + (1 - c) * TORCHIEKF.outer(axis, axis) \
                 + s * skew_axis).float()

        x = J.mm(xi[3:].view(-1, 3).t()).float()
        return Rot, x

    @staticmethod
    def so3exp(phi):
        angle = phi.norm().float()

        # Near phi==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).float()
            Xi = TORCHIEKF.Id3 + skew_phi
            return Xi.float()
        axis = phi / angle
        skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]]).float()
        c = angle.cos()
        s = angle.sin()
        Xi = c * TORCHIEKF.Id3 + (1 - c) * TORCHIEKF.outer(axis, axis) \
             + s * skew_axis
        return Xi.float()

    @staticmethod
    def outer(a, b):
        ab = a.view(-1, 1)*b.view(1, -1).float()
        return ab

    @staticmethod
    def so3left_jacobian(phi):
        angle = torch.norm(phi).float()

        # Near |phi|==0, use first order Taylor expansion
        if isclose(angle, 0.):
            skew_phi = torch.Tensor([[0, -phi[2], phi[1]],
                          [phi[2], 0, -phi[0]],
                          [-phi[1], phi[0], 0]]).float()
            return TORCHIEKF.Id3 + 0.5 * skew_phi

        axis = phi / angle
        skew_axis = torch.Tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]]).float()
        s = torch.sin(angle).float()
        c = torch.cos(angle).float()

        return ((s / angle) * TORCHIEKF.Id3 + (1 - s / angle) * TORCHIEKF.outer(axis, axis)\
               + ((1 - c) / angle) * skew_axis).float()

    @staticmethod
    def to_rpy(Rot):
        """Convert a rotation matrix to RPY Euler angles."""

        pitch = torch.atan2(-Rot[2, 0], torch.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2)).float()

        if isclose(pitch, np.pi / 2.):
            yaw = pitch.new_zeros(1).float()
            roll = torch.atan2(Rot[0, 1], Rot[1, 1]).float()
        elif isclose(pitch, -np.pi / 2.):
            yaw = pitch.new_zeros(1).float()
            roll = -torch.atan2(Rot[0, 1],  Rot[1, 1]).float()
        else:
            sec_pitch = 1. / pitch.cos()
            yaw = torch.atan2(Rot[1, 0] * sec_pitch, Rot[0, 0] * sec_pitch).float()
            roll = torch.atan2(Rot[2, 1] * sec_pitch, Rot[2, 2] * sec_pitch).float()
        return roll, pitch, yaw

    @staticmethod
    def from_rpy(roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles."""

        return TORCHIEKF.rotz(yaw).mm(TORCHIEKF.roty(pitch).mm(TORCHIEKF.rotx(roll))).float()

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""

        c = torch.cos(t).float()
        s = torch.sin(t).float()
        return t.new([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]]).float()

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""

        c = torch.cos(t).float()
        s = torch.sin(t).float()
        return t.new([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]]).float()

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""

        c = torch.cos(t).float()
        s = torch.sin(t).float()
        return t.new([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]]).float()

    @staticmethod
    def normalize_rot(rot):
        # U, S, V = torch.svd(A) returns the singular value
        # decomposition of a real matrix A of size (n x m) such that A=USV′.
        # Irrespective of the original strides, the returned matrix U will
        # be transposed, i.e. with strides (1, n) instead of (n, 1).

        # pytorch SVD seems to be inaccurate, so just move to numpy immediately
        U, _, V = torch.svd(rot).float()
        S = torch.eye(3).float()
        S[2, 2] = torch.det(U) * torch.det(V).float()
        return U.mm(S).mm(V.t()).float()

    def forward_nets(self, u):
        u_n = self.normalize_u(u).t().unsqueeze(0).float()
        u_n = u_n[:, :6].float()
        measurements_covs = self.mes_net(u_n, self).float()
        return measurements_covs

    def normalize_u(self, u):
        return ((u-self.u_loc)/self.u_std).float()

    def get_normalize_u(self, dataset):
        self.u_loc = dataset.normalize_factors['u_loc'].float()
        self.u_std = dataset.normalize_factors['u_std'].float()

    def set_Q(self):
        """
        Update the process noise covariance
        :return:
        """

        self.Q = torch.diag(torch.Tensor([self.cov_omega, self.cov_omega, self. cov_omega,
                                           self.cov_acc, self.cov_acc, self.cov_acc,
                                           self.cov_b_omega, self.cov_b_omega, self.cov_b_omega,
                                           self.cov_b_acc, self.cov_b_acc, self.cov_b_acc,
                                           self.cov_Rot_c_i, self.cov_Rot_c_i, self.cov_Rot_c_i,
                                           self.cov_t_c_i, self.cov_t_c_i, self.cov_t_c_i])
                            ).float()

        beta = self.initprocesscov_net.init_processcov(self)
        self.Q = torch.zeros(self.Q.shape[0], self.Q.shape[0]).float()
        self.Q[:3, :3] = self.cov_omega*beta[0]*self.Id3.float()
        self.Q[3:6, 3:6] = self.cov_acc*beta[1]*self.Id3.float()
        self.Q[6:9, 6:9] = self.cov_b_omega*beta[2]*self.Id3.float()
        self.Q[9:12, 9:12] = self.cov_b_acc*beta[3]*self.Id3.float()
        self.Q[12:15, 12:15] = self.cov_Rot_c_i*beta[4]*self.Id3.float()
        self.Q[15:18, 15:18] = self.cov_t_c_i*beta[5]*self.Id3.float()

    def load(self, args, dataset):
        path_iekf = os.path.join(args.path_temp, "iekfnets.p")
        if os.path.isfile(path_iekf):
            mondict = torch.load(path_iekf)
            self.load_state_dict(mondict)
            cprint("IEKF nets loaded", 'green')
        else:
            cprint("IEKF nets NOT loaded", 'yellow')
        self.get_normalize_u(dataset)


def isclose(mat1, mat2, tol=1e-10):
    return (mat1 - mat2).abs().lt(tol).float()


def prepare_filter(args, dataset):
    torch_iekf = TORCHIEKF()
    torch_iekf.load(args, dataset)
    torch_iekf = TORCHIEKF()

    # set dataset parameter
    torch_iekf.filter_parameters = args.parameter_class()
    torch_iekf.set_param_attr()
    if type(torch_iekf.g).__module__ == np.__name__:
        torch_iekf.g = torch.from_numpy(torch_iekf.g).float()

    # load model
    torch_iekf.load(args, dataset)
    torch_iekf.get_normalize_u(dataset)

    iekf = NUMPYIEKF(args.parameter_class)
    iekf.set_learned_covariance(torch_iekf)
    return iekf, torch_iekf
