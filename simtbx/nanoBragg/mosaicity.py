
from scipy import special
import numpy as np
from scitbx.matrix import sqr, col
import pylab as plt


d_eta_tensor_a = sqr((1, 0, 0,
                      0, 0, 0,
                      0, 0, 0))
d_eta_tensor_b = sqr((0, 0, 0,
                      0, 1, 0,
                      0, 0, 0))
d_eta_tensor_c = sqr((0, 0, 0,
                      0, 0, 0,
                      0, 0, 1))
d_eta_tensor_d = sqr((0, 1, 0,
                      1, 0, 0,
                      0, 0, 0))
d_eta_tensor_e = sqr((0, 0, 0,
                      0, 0, 1,
                      0, 1, 0))
d_eta_tensor_f = sqr((0, 0, 1,
                      0, 0, 0,
                      1, 0, 0))

d_eta_tensor_isotropic = sqr((1,0,0,
                              0,1,0,
                              0,0,1))


def search_directions(N=1000):
    """
    See Journal of Magnetic Resonance 138, 288–297 (1999)
    equation A6
    :param N: number of points on hemisphere
    :return: Nx3 numpy array of unit vectors
    """
    ti = (np.arange(1, N+1) - 0.5) / N
    THETA = np.arccos(ti)
    PHI = np.sqrt(np.pi*N) * np.arcsin(ti)

    u_vecs = np.zeros((N, 3))
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    u_vecs[:N, 0] = x
    u_vecs[:N, 1] = y
    u_vecs[:N, 2] = z

    return u_vecs


def generate_Umats(eta_tensor, crystal=None, num_axes=10, num_angles_per_axis=10, plot=False,
                   how=1):
    """

    :param crystal: dxtbx crystal model
    :param eta_tensor: sequence of 9 numbers specifying the mosacitiy tensor
    :param num_axes: how many points to sample the unit hemisphere with
    :param num_angles_per_axis:  produces 2x this number of angles per axes to sample the angle distribution
    :param plot: whether to visualize mosaicity cap by operating on the vector 1,0,0 with all Umats
    :param how: if 0, then do the full treatment (6 Umat derivatives)
                if 1, then do the diagonal only (3 Umat derivatives)
                if 2, then there is no anisotropy (1 Umat derivative)
    :return:
    """
    if plot:
        from mpl_toolkits.mplot3d import Axes3D
    if how == 1:
        for i in [1, 2, 3, 5, 6, 7]:
            assert eta_tensor[i] == 0
    elif how == 2:
        assert eta_tensor[0] == eta_tensor[4] == eta_tensor[8]
    else:
        assert eta_tensor[1] == eta_tensor[3]
        assert eta_tensor[2] == eta_tensor[6]
        assert eta_tensor[5] == eta_tensor[7]

    if how in [0, 1]:
        assert crystal is not None

    for val in eta_tensor:
        if val < 0:
            raise ValueError("Mosaicities need to be >= 0")

    if crystal is not None:
        a, b, c = map(col, crystal.get_real_space_vectors())
        unit_a = a.normalize()
        unit_b = b.normalize()
        unit_c = c.normalize()

    if plot:
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        f2  = plt.figure()
        ax2 = f2.add_subplot(111, projection='3d')

    hemisph_samples = search_directions(num_axes)

    if plot:
        x,y,z = hemisph_samples.T
        ax2.scatter(x,y,z,s=5,marker='s', color='r', alpha=0.5)
        A = col((1,0,0))
        all_anew = []

    eta_tensor = sqr(eta_tensor)

    if how == 0:  # full treatment
        derivs = [d_eta_tensor_a, d_eta_tensor_b, d_eta_tensor_c,
                  d_eta_tensor_d, d_eta_tensor_e, d_eta_tensor_f]
    elif how == 1:  # diag only treatment
        derivs = [d_eta_tensor_a, d_eta_tensor_b, d_eta_tensor_c]
    else:  # isotropic treatment
        derivs = [d_eta_tensor_isotropic]

    all_U = []
    all_Uprime = []
    all_Udblprime = []
    for i, pt in enumerate(hemisph_samples):
        rot_ax = col(pt)

        if crystal is not None:
            Ca = rot_ax.dot(unit_a)
            Cb = rot_ax.dot(unit_b)
            Cc = rot_ax.dot(unit_c)
            C = col((Ca, Cb, Cc)).normalize()
        else:
            C = col((1, 0, 0))  # arbitrary

        for ii in range(num_angles_per_axis):
            # position along y-axis of Cumulative distribution function
            ang_idx = float(ii) / num_angles_per_axis
            # effective mosaic rotation dependent on eta tensor
            eta_eff = C.dot(eta_tensor*C)

            # rotation amount
            factor = np.sqrt(2) * special.erfinv(ang_idx) * np.pi / 180.
            rot_ang = eta_eff * factor

            # first derivatives
            d_theta_d_etas = []
            for d_eta_tensor in derivs:
                d_theta_d_etas.append(C.dot(d_eta_tensor*C) * factor)

            # second deriv of theta w.r.t eta is 0!

            # even distribution of rotations:
            for rot_sign in [1, -1]:
                U = rot_ax.axis_and_angle_as_r3_rotation_matrix(rot_sign*rot_ang, deg=False)
                all_U.append(U)

                dU_d_theta = rot_ax.axis_and_angle_as_r3_derivative_wrt_angle(rot_sign*rot_ang, deg=False)
                d2U_d_theta2 = rot_ax.axis_and_angle_as_r3_second_derivative_wrt_angle(rot_sign*rot_ang, deg=False)
                for d_theta_d_eta in d_theta_d_etas:
                    dU_d_eta = rot_sign*dU_d_theta*d_theta_d_eta
                    all_Uprime.append(dU_d_eta)

                    d2U_d_eta2 = d2U_d_theta2*d_theta_d_eta**2
                    all_Udblprime.append( d2U_d_eta2)

                if plot:
                    anew = U*A
                    all_anew.append(anew)

    if plot:
        x,y,z = np.array(all_anew).T
        ax.scatter(x,y,z,s=2, alpha=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    if how == 0:
        assert 6*len(all_U) == len(all_Uprime) == len(all_Udblprime)
    elif how == 1:
        assert 3 * len(all_U) == len(all_Uprime) == len(all_Udblprime)
    elif how == 0:
        assert len(all_U) == len(all_Uprime) == len(all_Udblprime)

    return all_U, all_Uprime, all_Udblprime


if __name__ == "__main__":
    cryst_dict = dict([('__id__', 'crystal'), ('real_space_a', (-48.93914505851325, -61.4985726090971, 0.23980318971727585)), ('real_space_b', (-27.63556200961052, 72.26768337463876, 13.81410546001183)), ('real_space_c', (-42.92524538136074, 33.14788397044063, -259.2845460893375)), ('space_group_hall_symbol', '-P 6 2'), ('ML_half_mosaicity_deg', 0.02676231907923616), ('ML_domain_size_ang', 4646.073492432425)])
    from dxtbx.model import Crystal
    C = Crystal.from_dict(cryst_dict)

    # mosaicities in degrees
    a, b, c = 0.025, 0.025, 0.075
    d, e, f = 0.01, 0.05, 0.09

    # spherical cap model
    etas = (a, 0, 0,
            0, a, 0,
            0, 0, a)
    U, _, _ =generate_Umats(etas,C, num_axes=200, num_angles_per_axis=20, plot=True, how=2)
    # isotropic case should not depend on the crystal model
    U2, _, _ =generate_Umats(etas,None, num_axes=200, num_angles_per_axis=20, plot=False, how=2)
    for i in range(len(U)):
        assert np.allclose(U[i].elems, U2[i].elems)

    # anisotropic cap model
    etas = (a, 0, 0,
            0, b, 0,
            0, 0, c)
    generate_Umats(etas, C, num_axes=200, num_angles_per_axis=20, plot=True, how=1)

    # fully anisotropic cap model
    etas = (a, d, f,
            d, b, e,
            f, e, c)
    generate_Umats(etas, C, num_axes=200, num_angles_per_axis=20, plot=True, how=0)
