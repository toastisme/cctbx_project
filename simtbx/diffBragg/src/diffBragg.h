
#include <simtbx/nanoBragg/nanoBragg.h>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include <Eigen/Dense>
#include <boost/python.hpp>
#include<Eigen/StdVector>
#include <boost/python/numpy.hpp>
typedef std::vector<double> image_type;

#ifdef NANOBRAGG_HAVE_CUDA
#include "diffBraggCUDA.h"
#endif

//#include <boost/python/numpy.hpp>

namespace simtbx {
namespace nanoBragg {

class derivative_manager{
  public:
    virtual ~derivative_manager(){}
    derivative_manager();
    void initialize(int Npix_total, bool compute_curvatures);
    af::flex_double raw_pixels;
    af::flex_double raw_pixels2;
    double value; // the value of the parameter
    double dI; // the incremental derivative
    double dI2; // the incremental derivative
    bool refine_me;
    bool second_derivatives;
    void increment_image(int idx, double value, double value2, bool curvatures);

};

class panel_manager: public derivative_manager{
  public:
    panel_manager();
   virtual void foo(){}
    virtual ~panel_manager(){}
    //virtual void set_R();
    //void increment(double Iincrement, double omega_pixel, Eigen::Matrix3d M, double pix2, Eigen::Vector3d o, Eigen::Vector3d k_diffracted,
    //            double per_k, double per_k3, double per_k5, Eigen::Vector3d V);

    Eigen::Matrix3d dR;
    Eigen::Vector3d dF;
    Eigen::Vector3d dS;
    Eigen::Vector3d dk;
    Eigen::Vector3d F_cross_dS;
    Eigen::Vector3d dF_cross_S;

}; // end of rot_manager


class rot_manager: public derivative_manager{
  public:
    rot_manager();
    virtual ~rot_manager(){}
    virtual void set_R();
    void increment(double value, double value2);
    Eigen::Matrix3d XYZ, XYZ2;
    Eigen::Matrix3d R, dR, dR2;

}; // end of rot_manager

class Ncells_manager: public derivative_manager{
  public:
    Ncells_manager();
    virtual ~Ncells_manager(){}
    void increment(double dI_increment, double dI2_increment);
    //bool single_parameter;  // refine a single parameter (Na=Nb=Nc), or else all 3 parameters (Na, Nb, Nc)
};


class Fcell_manager: public derivative_manager{
  public:
    Fcell_manager();
    virtual ~Fcell_manager(){}
    void increment(double value, double value2);
};

class eta_manager: public derivative_manager{
  public:
    eta_manager();
    virtual ~eta_manager(){}
    void increment(double value, double value2);
};

class lambda_manager: public derivative_manager{
  public:
    lambda_manager();
    virtual ~lambda_manager(){}
    void increment(double value, double value2);
    double dg_dlambda;
};


class ucell_manager: public derivative_manager{
  public:
    ucell_manager();
    virtual ~ucell_manager(){}
    void increment(double value, double value2);

    Eigen::Matrix3d dB, dB2;
};

class origin_manager: public derivative_manager{
  public:
    origin_manager();
    virtual ~origin_manager(){}
    //void increment(
    //    const Eigen::Ref<const Eigen::Vector3d>vec3 V,
    //    const Eigen::Ref<const Eigen::Matrix3d>mat3 N, const Eigen::Ref<const Eigen::Matrix3d>mat3 UBO,
    //    const Eigen::Ref<const Eigen::Vector3d>vec3 k_diffracted, const Eigen::Ref<const Eigen::Vector3d>vec3 o_vec,
    //    double air_path, double wavelen, double Hrad, double Fcell, double Flatt, double fudge,
    //    double source_I, double capture_fraction, double omega_pixel, double pixel_size);
    Eigen::Vector3d dk; /* derivative of the diffracted vector along this origin component */
    double FF, FdF, FdF2, dFdF;
};

class rotX_manager: public rot_manager{
  public:
    rotX_manager();
    void set_R();
};
class rotY_manager: public rot_manager{
  public:
    rotY_manager();
    void set_R();
};
class rotZ_manager: public rot_manager{
  public:
    rotZ_manager();
    void set_R();
};

class diffBragg: public nanoBragg{
  public:
  diffBragg(const dxtbx::model::Detector& detector, const dxtbx::model::Beam& beam,
            int verbose);

  void diffBragg_sum_over_steps(
        int Npix_to_model, std::vector<unsigned int>& panels_fasts_slows,
        image_type& floatimage,
        image_type& d_Umat_images, image_type& d2_Umat_images,
        image_type& d_Bmat_images, image_type& d2_Bmat_images,
        image_type& d_Ncells_images, image_type& d2_Ncells_images,
        image_type& d_fcell_images, image_type& d2_fcell_images,
        image_type& d_eta_images,
        image_type& d2_eta_images,
        image_type& d_lambda_images, image_type& d2_lambda_images,
        image_type& d_panel_rot_images, image_type& d2_panel_rot_images,
        image_type& d_panel_orig_images, image_type& d2_panel_orig_images,
        image_type& d_sausage_RXYZ_scale_images,
        image_type& d_fp_fdp_images,
        int* subS_pos, int* subF_pos, int* thick_pos,
        int* source_pos, int* phi_pos, int* mos_pos, int* sausage_pos,
        const int Nsteps, int _printout_fpixel, int _printout_spixel, bool _printout, double _default_F,
        int oversample, bool _oversample_omega, double subpixel_size, double pixel_size,
        double detector_thickstep, double _detector_thick, std::vector<double>& close_distances, double detector_attnlen,
        bool use_lambda_coefficients, double lambda0, double lambda1,
        Eigen::Matrix3d& eig_U, Eigen::Matrix3d& eig_O, Eigen::Matrix3d& eig_B, Eigen::Matrix3d& RXYZ,
        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& dF_vecs,
        std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> >& dS_vecs,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& UMATS_RXYZ,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& UMATS_RXYZ_prime,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& UMATS_RXYZ_dbl_prime,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& RotMats,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& dRotMats,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& d2RotMats,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& UMATS,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& dB_Mats,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& dB2_Mats,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& sausages_RXYZ,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& d_sausages_RXYZ,
        std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> >& sausages_U,
        std::vector<double>& sausages_scale,
        double* source_X, double* source_Y, double* source_Z, double* source_lambda, double* source_I,
        double kahn_factor,
        double Na, double Nb, double Nc,
        double Nd, double Ne, double Nf,
        double phi0, double phistep,
        Eigen::Vector3d& spindle_vec, Eigen::Vector3d& _polarization_axis,
        int h_range, int k_range, int l_range,
        int h_max, int h_min, int k_max, int k_min, int l_max, int l_min, double dmin,
        double fudge, bool complex_miller, int verbose, bool only_save_omega_kahn,
        bool isotropic_ncells, bool compute_curvatures,
        std::vector<double>& _FhklLinear, std::vector<double>& _Fhkl2Linear,
        std::vector<bool>& refine_Bmat, std::vector<bool>& refine_Ncells, bool refine_Ncells_def,  std::vector<bool>& refine_panel_origin, std::vector<bool>& refine_panel_rot,
        bool refine_fcell, std::vector<bool>& refine_lambda, bool refine_eta, std::vector<bool>& refine_Umat,
        bool refine_sausages,
        int num_sausages,
        bool refine_fp_fdp,
        std::vector<double>& fdet_vectors, std::vector<double>& sdet_vectors,
        std::vector<double>& odet_vectors, std::vector<double>& pix0_vectors,
        bool _nopolar, bool _point_pixel, double _fluence, double _r_e_sqr, double _spot_scale,
        bool no_Nabc_scale,
        std::vector<double>& fpfdp,
        std::vector<double>& fpfdp_derivs,
        std::vector<double>& atom_data, bool track_Fhkl, std::vector<int>& nominal_l);


  bool track_Fhkl;
  std::vector<int> nominal_l;
  void update_xray_beams(scitbx::af::versa<dxtbx::model::Beam, scitbx::af::flex_grid<> > const& value);
  void diffBragg_rot_mats();
  void linearize_Fhkl();
  void sanity_check_linear_Fhkl();
  void update_linear_Fhkl();
  void diffBragg_list_steps(
                int* subS_pos,  int* subF_pos,  int* thick_pos,
                int* source_pos,  int* phi_pos,  int* mos_pos , int* sausage_pos);
  ~diffBragg(){};
  void diffBragg_sum_over_steps_cuda();

#ifdef NANOBRAGG_HAVE_CUDA
    diffBragg_cudaPointers device_pointers;
    inline void gpu_free(){
        freedom(device_pointers);
    }
    
#endif

  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > dF_vecs, dS_vecs;
  void initialize_managers();
  void vectorize_umats();
  void rotate_fs_ss_vecs(double panel_rot_ang);
  void rotate_fs_ss_vecs_3D(double panel_rot_angO, double panel_rot_angF, double panel_rot_angS);
  void add_diffBragg_spots(const af::shared<size_t>& panels_fasts_slows);
  void add_diffBragg_spots(const af::shared<size_t>& panels_fasts_slows, boost::python::list per_pix_nominal_l);
  void add_diffBragg_spots();
  void init_raw_pixels_roi();
  void zero_raw_pixel_rois();
  void set_ucell_derivative_matrix(int refine_id, af::shared<double> const& value);
  void set_ucell_second_derivative_matrix(int refine_id, af::shared<double> const& value);
  void init_Fhkl2();
  af::flex_double get_panel_increment(double Iincrement, double omega_pixel, const Eigen::Ref<const Eigen::Matrix3d>& M, double pix2,
            const Eigen::Ref<const Eigen::Vector3d>& o, const Eigen::Ref<const Eigen::Vector3d>& k_diffracted,
            double per_k, double per_k3, double per_k5, const Eigen::Ref<const Eigen::Vector3d>& V, const Eigen::Ref<const Eigen::Vector3d>& _dk);
  inline void free_Fhkl2(){
      if(Fhkl2 != NULL) {
        for (h0=0; h0<=h_range;h0++) {
          for (k0=0; k0<=k_range;k0++) {
            free(Fhkl2[h0][k0]);
          }
          free(Fhkl2[h0]);
        }
        free(Fhkl2);
      }
  }
  //void reset_derivative_pixels(int refine_id);

  /* methods for interacting with the derivative managers */
  void refine(int refine_id);
  void fix(int refine_id);
  void let_loose(int refine_id);
  int detector_panel_id;
  void shift_originZ(const dxtbx::model::Detector& detector, double shift);
  void update_dxtbx_geoms(const dxtbx::model::Detector& detector, const dxtbx::model::Beam& beam,
        int panel_id, double panel_rot_angO=0,
        double panel_rot_angF=0,  double panel_rot_angS=0, double panel_offsetX=0,
        double panel_offsetY=0, double panel_offsetZ=0, bool force=true);
  void set_value( int refine_id, double value);
  void set_sausages(af::flex_double x, af::flex_double y, af::flex_double z, af::flex_double scale);
  void set_ncells_values( boost::python::tuple const& values);
  void set_ncells_def_values( boost::python::tuple const& values);
  void print_if_refining();
  boost::python::tuple get_ncells_values();
  boost::python::tuple get_fcell_derivative_pixels();
  boost::python::list get_sausage_derivative_pixels();
  boost::python::list get_sausage_scale_derivative_pixels();
  bool refining_sausages=false;
  double get_value( int refine_id);
  af::flex_double get_derivative_pixels(int refine_id);
  af::flex_double get_second_derivative_pixels(int refine_id);
  af::flex_double get_raw_pixels_roi();
  boost::python::tuple get_fp_fdp_derivative_pixels();
  boost::python::tuple get_ncells_derivative_pixels();
  boost::python::numpy::ndarray get_Na_derivative_pixels();
  boost::python::tuple get_ncells_def_derivative_pixels();
  boost::python::tuple get_ncells_def_second_derivative_pixels();
  boost::python::tuple get_ncells_second_derivative_pixels();
  boost::python::tuple get_aniso_eta_deriv_pixels();
  boost::python::tuple get_aniso_eta_second_deriv_pixels();

  boost::python::tuple get_lambda_derivative_pixels();

  /* override to cache some of the polarization calc variables to use in derivatives*/
  double polarization_factor(double kahn_factor, double *incident, double *diffracted, double *axis);

  double psi;  /* for polarization correction */
  double u;
  double kEi;
  double kBi;
  Eigen::Vector3d O_reference;

  //bool use_omega_pixel_ave;
  double om;
  double omega_pixel_ave;
  double airpath_ave;

  double diffracted_ave[4];
  double pixel_pos_ave[4];
  double Fdet_ave, Sdet_ave, Odet_ave;
  Eigen::Vector3d k_diffracted_ave;
  Eigen::Vector3d k_incident_ave;

  Eigen::Matrix3d EYE;
  mat3 Umatrix;
  mat3 Bmatrix;
  mat3 Omatrix;
  Eigen::Matrix3d NABC;
  Eigen::Matrix3d RXYZ;
  std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > Fdet_vectors, Sdet_vectors, Odet_vectors;
  std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > RotMats, dRotMats, d2RotMats, R3, R3_2,
    UMATS, UMATS_RXYZ, UMATS_prime, UMATS_dbl_prime,UMATS_RXYZ_prime, UMATS_RXYZ_dbl_prime;
  //std::vector<Eigen::Matrix3d> RotMats;
  //std::vector<Eigen::Matrix3d> dRotMats, d2RotMats;
  //std::vector<Eigen::Matrix3d> R3, R3_2;

  // Panel rotation
  Eigen::Matrix3d panR;
  Eigen::Matrix3d panR2;
  double panel_rot_ang;

  //vec3 k_diffracted;
  //vec3 o_vec;
  Eigen::Vector3d Ei_vec, Bi_vec;
  //vec3 H_vec, H0_vec;
  //vec3 a_vec, ap_vec;
  //vec3 b_vec, bp_vec;
  //vec3 c_vec, cp_vec;
  //vec3 q_vec; // scattering vector

  //std::vector<Eigen::Matrix3d> UMATS;
  //std::vector<Eigen::Matrix3d> UMATS_RXYZ;
  //std::vector<Eigen::Matrix3d> UMATS_prime;
  //std::vector<Eigen::Matrix3d> UMATS_RXYZ_prime;
  double * mosaic_umats_prime;
  int nmats; // the number of mosaic umat derivative matrices (can be 3x the number of mosaic domains)

  double * mosaic_umats_dbl_prime;
  void set_mosaic_blocks_prime(af::shared<mat3> umat_in);  // set the individual mosaic block orientation derivatives from array
  void set_mosaic_blocks_dbl_prime(af::shared<mat3> umat_in);  // set the individual mosaic block orientation derivatives from array
  //bool vectorized_umats;

  /* derivative managers */
  std::vector<boost::shared_ptr<Fcell_manager> > fp_fdp_managers;
  std::vector<boost::shared_ptr<rot_manager> > sausage_managers;
  std::vector<boost::shared_ptr<derivative_manager> > sausage_scale_managers;
  std::vector<boost::shared_ptr<rot_manager> > rot_managers;
  std::vector<boost::shared_ptr<ucell_manager> > ucell_managers;
  std::vector<boost::shared_ptr<Ncells_manager> > Ncells_managers;
  std::vector<boost::shared_ptr<eta_manager> > eta_managers;
  std::vector<boost::shared_ptr<origin_manager> > origin_managers;
  std::vector<boost::shared_ptr<lambda_manager> > lambda_managers;
  std::vector<boost::shared_ptr<panel_manager> > panels;
  std::vector<boost::shared_ptr<derivative_manager> > fcell_managers;
  //boost::shared_ptr<eta_manager> eta_man;
  boost::shared_ptr<panel_manager> panel_rot_man;
  boost::shared_ptr<panel_manager> panel_rot_manF;
  boost::shared_ptr<panel_manager> panel_rot_manS;

  double* floatimage_roi;
  af::flex_double raw_pixels_roi;
  //af::flex_int raw_pixels_roi;
  //af::flex_double raw_pixels_roi;
  //af::flex_double raw_pixels_roi;

  bool compute_curvatures;
  bool update_oversample_during_refinement;
  bool oversample_omega;
  bool only_save_omega_kahn;
  double uncorrected_I;
  Eigen::Vector3d max_I_hkl;// the hkl corresponding to the maximum intensity in the array (debug)
  //int max_I_h, max_I_k, max_I_l;

  // helpful definitions..
  double per_k ;
  double per_k2 ;
  double per_k3 ;
  double per_k4 ;
  double per_k5;
  double per_k6;
  double per_k7;
  double G ;

  double du;
  double du2;

  double w ;
  double w2;
  double BperE2;
  double dkE ;
  double dkB;
  double v ;
  double dv;
  double dpsi ;
  double dpsi2;

  double c2psi ;
  double s2psi ;
  double gam_cos2psi;
  double gam_sin2psi;

  double dpolar ;

  double dpolar2;

  double pp ;
  double dOmega ;
  double dOmega2;

  double FF ;
  double FdF ;
  double dFdF ;
  double FdF2;

  /* om is the average solid angle in the pixel (average over sub pixels) */
  double origin_dI;
  double origin_dI2;
  // different scale term here because polar and domega terms depend on originZ
  double scale_term2;
  double scale_term;

  void quick_Fcell_update(boost::python::tuple const& value);
  double ***Fhkl2;  // = NULL
  af::shared<double> pythony_amplitudes2;
  bool complex_miller;
  double F_cell2; // for storing the imaginary component
  std::vector<double> fpfdp;
  std::vector<double> fpfdp_derivs;
  std::vector<double> atom_data;
  void show_heavy_atom_data();
  void show_fp_fdp();

  bool isotropic_ncells;
  bool modeling_anisotropic_mosaic_spread;
  double Nd, Ne, Nf;
  bool refine_Ncells_def;
  bool no_Nabc_scale;  // if true, then absorb the Nabc scale into an overall scale factor

  double source_lambda0, source_lambda1;
  bool use_lambda_coefficients;
  std::vector<double> FhklLinear, Fhkl2Linear;
  // Eigen types
  Eigen::Matrix3d eig_U, eig_B, eig_O;
  std::vector<double> fdet_vectors, sdet_vectors, pix0_vectors, odet_vectors, close_distances;
  void set_close_distances();
  int Npix_total, Npix_to_model;

  // cuda properties
  bool update_dB_matrices_on_device=false;
  bool update_detector_on_device=false;
  bool update_rotmats_on_device=false;
  bool update_sausages_on_device=false;
  bool update_umats_on_device=false;
  bool update_panels_fasts_slows_on_device=false;
  bool update_sources_on_device=false;
  bool update_Fhkl_on_device=false;
  bool update_refine_flags_on_device=false;
  bool update_step_positions_on_device=false;
  bool update_panel_deriv_vecs_on_device=false;
  bool use_cuda=false;
  int Npix_to_allocate=-1; // got GPU allocation, -1 is auto mode

  void update_number_of_sausages(int _num_sausages);
  void allocate_sausages();
  int num_sausages=1;
  std::vector<Eigen::Matrix3d,Eigen::aligned_allocator<Eigen::Matrix3d> > sausages_RXYZ, d_sausages_RXYZ, sausages_U;
  std::vector<double> sausages_scale;
}; // end of diffBragg

} // end of namespace nanoBragg
} // end of namespace simtbx
