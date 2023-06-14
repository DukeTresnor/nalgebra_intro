extern crate nalgebra as na;

use na::matrix;
//use na::Unit;
use na::Dyn;
use na::U6;

pub const TEST_THETA: f64 = std::f64::consts::PI * 1.0;

//use na::{U2, U3, Dynamic, ArrayStorage, VecStorage, Matrix};

// For logging progress, use a check mark with rust inside the text book -- check_rust 

// maybe i can make a function def using na::base::Matrix ..
// you forgot to include the type of the elements inside the matrix... do -- na::MatrixX<f64>
// make sure you go through this an implement generics, so you can input integers or floats into the functions!
// also be sure to wrap each function in options, so that you can catch any potential errors

fn main() {
    
    let _mat1 = na::Matrix4::new(
        2.0, 1.0, 1.0, -5.0,
        1.0, 2.0, 1.0, -9.4, 
        1.0, 1.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    

    //let vec1 = na::Vector3::new(4.0, -3.0, 1.0);

    //let mat2 = scew_symmetric_matrix_so3_from_vector_r3(&vec1);

    //println!("mat2: {}", mat2);
    //println!("vec1: {}", vec1);

    //let theta = TEST_THETA;


    //let mat3 = exponential_rotation_matrix_bigso3_from_exponential_coordinates_r3(&vec1, theta);

    //println!("rotation 3: {}", mat3);


    //let vec4 = vector_r3_from_scew_symmetric_matrix_so3(&mat2);

    //println!("vec4: {}", vec4);


    let logarithm_test = matrix![
        0.0, -1.0, 0.0;
        1.0, 0.0, 0.0;
        0.0, 0.0, 1.0;
    ];

    let logarithm_output_for_logarithm_test = matrix_logarithm_so3_from_rotation_matrix_bigso3(&logarithm_test);


    println!("logarithm_output_for_logarithm_test: {}", logarithm_output_for_logarithm_test);

    let unscewed_test = vector_r3_from_scew_symmetric_matrix_so3(&logarithm_output_for_logarithm_test);

    println!("unscewed vector from logarithm output: {}", unscewed_test);

    let exp_coor_test = exponential_rotation_matrix_bigso3_from_exponential_coordinates_so3(&logarithm_output_for_logarithm_test);

    println!("exp_coor_test: {}", exp_coor_test);



    // unit_vector_r3_angle_from_exponential_coordinates_r3

    let (omega_hat, theta_angle) = unit_vector_r3_angle_from_exponential_coordinates_r3(&unscewed_test);

    println!("omega_hat: {}, theta_angle: {}", omega_hat, theta_angle);


//transform_bigse3_from_rotation_bigso3_position_r3 
    let rotation_test = matrix![
        0.0, -1.0, 0.0;
        1.0, 0.0, 0.0;
        0.0, 0.0, 1.0;
    ];
    
    let position_test = na::Vector3::new(0.0, 0.0, 0.0);

    let transform_test = transform_bigse3_from_rotation_bigso3_position_r3(&rotation_test, &position_test);

    println!("transform_test: {}", transform_test);



//rotation_bigso3_position_r3_from_transform_bigse3
    let (rotation_test_2, position_test_2) = rotation_bigso3_position_r3_from_transform_bigse3(&transform_test);
    println!("rotation_test_2: {}, position_test_2: {}", rotation_test_2, position_test_2);


}

// -- Helper Functions -- //



fn near_zero(
    // params: A scalar input to check
    // returns: boolean, true if input is close to zero, false otherwise
    scalar_input: &f64,
) -> bool {
    //
    scalar_input.abs() < 1.0e-6
}

// Not returning option because im making the program panic?
// this should normalize the vector if it isn't normal yet
fn check_for_normalization_r3(
    //
    vector_r3: &na::Vector3<f64>,
) -> na::Vector3<f64> {
    if let Some(normal_vector_r3) = vector_r3.try_normalize(1.0e-6) {
        normal_vector_r3
    }
    else {
        panic!("Not a normalized vector")
        //println!("Not a normalized vector -- normalizing");
        //vector_r3.normalize()
    }
}

// adapt to make it generic on any square matrix
fn trace_of_matrix3(
    // params: any 3x3 matrix with f64 as elements
    // returns: the trace of the input matrix, ie the sum of the diagonal elements
    matrix3by3: &na::Matrix3<f64>
) -> f64{
    let trace = matrix3by3[(0,0)] + matrix3by3[(1,1)] + matrix3by3[(2,2)];
    println!("trace of {}: {}", matrix3by3, trace);
    trace
}

fn unit_vector_r3_angle_from_exponential_coordinates_r3(
    // params: exponential_coordinates -- 3-vector in R3 representing the exponential coordinates necessary for a particular rotation
    // returns: omega_hat -- a unit rotation axis
    // returns: theta_angle -- the rotation angle coresponding to the given exponential coordinates
    exponential_coordinates: &na::Vector3::<f64>,
) -> (na::Vector3<f64>, f64) {
    //
    let omega_hat = check_for_normalization_r3(&exponential_coordinates);
    let theta_angle = exponential_coordinates.norm();

    (omega_hat, theta_angle)
}



// -- Helper Functions -- //







// -- Chapter 3 -- //

fn scew_symmetric_matrix_so3_from_vector_r3(
    // params: Any 3-vector in R3 as a vector f64's-- ex: (x, y, z)
    // returns: skew symmetric matrix [w] in so(3)
    vector_r3: &na::Vector3<f64>,

) -> na::Matrix3<f64> {
    //

    let scew_mat = matrix![
        0.0          , -vector_r3[2], vector_r3[1] ;
        vector_r3[2] , 0.0          , -vector_r3[0];
        -vector_r3[1], vector_r3[0] , 0.0          ;
    ];

    scew_mat
}

fn vector_r3_from_scew_symmetric_matrix_so3(
    // params: skew symmetric matrix [w] in so(3)
    // returns: A 3-vector representation of [w], w, in R3 -- x, y, z in f64
    screw_symmetric_matrix: &na::Matrix3<f64>,
) -> na::Vector3<f64> {
    
    let vector_r3_x = screw_symmetric_matrix[(2, 1)];
    let vector_r3_y = screw_symmetric_matrix[(0, 2)];
    let vector_r3_z = screw_symmetric_matrix[(1, 0)];


    let vector_r3 = na::Vector3::new(vector_r3_x, vector_r3_y, vector_r3_z);
    vector_r3
}

// this function should eventually take an input of [w]theta , and work on extracting [w] and theta from that input in addition to outputing the rotation matrix
fn exponential_rotation_matrix_bigso3_from_exponential_coordinates_so3(
    // params: Any normalized 3-vector in R3 representing angular velocity (?) w is in R3
    //           should be able to wrap the vector input into a Unit to say it needs to be a Unit Vector type, but not sure how to do at the moment
    // params: Any scalar theta representing the amoung of angular rotation
    // patams: skew_exponential_coordinates: a 3x3 skew-symmetric matrix representing exponential coordinates
    // returns: The rotation matrix R in SO3 resulting from exponential coordinates w_theta
    // old coordinates, now we only give skewed exponential coordinates -- ie matrix in, matrix out
    //vector_r3: &na::Vector3<f64>,
    //theta_scalar: f64,
    skew_exponential_coordinates: &na::Matrix3<f64>,

) -> na::Matrix3<f64> {
    //


    let un_scewed_exponential_coordinates = vector_r3_from_scew_symmetric_matrix_so3(&skew_exponential_coordinates);

    if near_zero(&un_scewed_exponential_coordinates.norm()) {
        return matrix![
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
        ]
    } else {
        let (rotation_axis, theta_angle) = unit_vector_r3_angle_from_exponential_coordinates_r3(&un_scewed_exponential_coordinates);

        let normal_vector_r3 = check_for_normalization_r3(&rotation_axis);

        let scew_exponential_coordinates = scew_symmetric_matrix_so3_from_vector_r3(&normal_vector_r3);

        let scew_exponential_coordinates_squared = scew_exponential_coordinates * scew_exponential_coordinates;

        let rotation: na::Matrix3<f64> = na::Matrix3::identity() + theta_angle.sin() * scew_exponential_coordinates + (1.0 - theta_angle.cos()) * scew_exponential_coordinates_squared;

        return rotation
    }




}



fn matrix_logarithm_so3_from_rotation_matrix_bigso3(
    // params: 3x3 rotation matrix in SO3
    // returns: the matrix logarithm of R, which is a 3x3 skew-symmetric vector [w]theta -- or [omega] * theta_value
    rotation_matrix_bigso3: &na::Matrix3<f64>
) -> na::Matrix3<f64> {
    //

    let mut omega_vector = na::Vector3::new(0.0, 0.0, 0.0);

    let mut theta_value = 0.0;

    let dummy_return = matrix![
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
    ];

    // use the trace to extract the angle of rotation from the input rotation matrix
    // replace with match case later
    let cosine_of_theta = (trace_of_matrix3(&rotation_matrix_bigso3) - 1.0) / 2.0;
    if cosine_of_theta >= 1.0 {
        println!("More than 1");
        println!("Theta is 0, and omega is undefined");
        // A little sloppy
        return dummy_return

    } else if cosine_of_theta <= -1.0 {
        println!("less than -1");
        if !near_zero(&(1.0 + rotation_matrix_bigso3[(2,2)])) {
            omega_vector = ( 1.0 / (2.0 * (1.0 + rotation_matrix_bigso3[(2,2)].sqrt() ) ) ) * 
                na::Vector3::new(
                    rotation_matrix_bigso3[(0,2)], rotation_matrix_bigso3[(1,2)], 1.0 + rotation_matrix_bigso3[(2,2)]
                );
            
        } else if !near_zero(&(1.0 + rotation_matrix_bigso3[(1,1)])) {
            omega_vector = ( 1.0 / (2.0 * (1.0 + rotation_matrix_bigso3[(1,1)].sqrt() ) ) ) * 
                na::Vector3::new(
                    rotation_matrix_bigso3[(0,1)], 1.0 + rotation_matrix_bigso3[(1,1)], rotation_matrix_bigso3[(2,1)]
                );
        } else {
            omega_vector = ( 1.0 / (2.0 * (1.0 + rotation_matrix_bigso3[(1,1)].sqrt() ) ) ) * 
                na::Vector3::new(
                    1.0 + rotation_matrix_bigso3[(0,0)], rotation_matrix_bigso3[(1,0)], rotation_matrix_bigso3[(2,0)]
                );
        }
        // in this case theta should be PI
        theta_value = std::f64::consts::PI;
        return scew_symmetric_matrix_so3_from_vector_r3(&omega_vector) * theta_value
    } else {
        // in this case theta should be the cosine inverse of the cosine_of_theta value
        println!("between -1 and 1");
        theta_value = cosine_of_theta.acos();
        let scew_omega = (1.0 / (2.0 * theta_value.sin())) * (rotation_matrix_bigso3 - rotation_matrix_bigso3.transpose());
        return scew_omega * theta_value

    }

    
}
/*
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
 */


fn transform_bigse3_from_rotation_bigso3_position_r3(
    // params: rotation_matrix -- A 3x3 rotation matrix in SO3 -- R
    // params: position_vector -- A 3x1 position vector in R3 -- p
    // returns: transform_matrix -- A 4x4 homogenous transform matrix in SE3, in the form T = [R, p; 0, 1] -- T
    rotation_matrix: &na::Matrix3<f64>,
    position_vector: &na::Vector3<f64>,
) -> na::Matrix4<f64> {
    //



    let transform_matrix = na::Matrix4::new(
        rotation_matrix[(0,0)], rotation_matrix[(0,1)], rotation_matrix[(0,2)], position_vector[0],
        rotation_matrix[(1,0)], rotation_matrix[(1,1)], rotation_matrix[(1,2)], position_vector[1], 
        rotation_matrix[(2,0)], rotation_matrix[(2,1)], rotation_matrix[(2,2)], position_vector[2],
        0.0                   , 0.0                   , 0.0                   , 1.0                   ,
    );

    transform_matrix
}


fn rotation_bigso3_position_r3_from_transform_bigse3(
    // params: transform_matrix -- A 4x4 homogenous transform matrix in SE3, in the form T = [R, p; 0, 1] -- T -- technically doesn't need to be homogeneous...
    // returns: rotation_matrix -- A 3x3 rotation matrix in SO3 -- R
    // returns: position_vector -- A 3x1 position vector in R3 -- p
    transform_matrix: &na::Matrix4<f64>,
) -> (na::Matrix3<f64>, na::Vector3<f64>) {
    //
    let rotation_matrix = matrix![
        transform_matrix[(0,0)], transform_matrix[(0,1)], transform_matrix[(0,2)];
        transform_matrix[(1,0)], transform_matrix[(1,1)], transform_matrix[(1,2)];
        transform_matrix[(2,0)], transform_matrix[(2,1)], transform_matrix[(2,2)];
    ];



    let position_vector = na::Vector3::new(
        transform_matrix[(0,3)], transform_matrix[(1,3)], transform_matrix[(2,3)]
    );

    (rotation_matrix, position_vector)

}


fn _efficient_homogeneous_matrix_inverse(
    // fill in later
) {
    //
}



fn twist_matrix_se3_from_twist_r6(
    // params: twist_r6 -- A 6-vector representing a spatial velocity -- twist = [w, v], with w being angular velocity, and v being linear velocity
    // returns: twist_matrix_se3 -- A 4x4 matrix representation of the 6-vector twist, in SE3 (not homogeneous)
    twist_r6: &na::Vector6<f64>,
) -> na::Matrix4<f64>{
    // converts first 3 elements of twist_r6 into skew symetric matrix, uses that as rotation in 4x4, last 3 twist elements are the position

    let rotation_from_twist = scew_symmetric_matrix_so3_from_vector_r3(&na::Vector3::new(twist_r6[0], twist_r6[1], twist_r6[2]));

    let twist_matrix_se3 = na::Matrix4::new(
        rotation_from_twist[(0,0)], rotation_from_twist[(0,1)], rotation_from_twist[(0,2)], twist_r6[3],
        rotation_from_twist[(1,0)], rotation_from_twist[(1,1)], rotation_from_twist[(1,2)], twist_r6[4], 
        rotation_from_twist[(2,0)], rotation_from_twist[(2,1)], rotation_from_twist[(2,2)], twist_r6[5],
        0.0                   , 0.0                   , 0.0                   , 0.0                   ,
    );

    twist_matrix_se3
}

/*

    """ Converts an se3 matrix into a spatial velocity vector

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

 */

fn twist_r6_from_twist_matrix_se3(
    // params: twist_matrix_se3 -- A 4x4 matrix representation of the 6-vector twist, in SE3 (not homogeneous)
    // returns: twist_r6 -- A 6-vector representing a spatial velocity -- twist = [w, v], with w being angular velocity, and v being linear velocity
    twist_matrix_se3: &na::Matrix4<f64>,
) -> na::Vector6<f64> {
    //     return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
    //             [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

    let twist_r6 = na::vector![twist_matrix_se3[(2,1)], twist_matrix_se3[(0,2)], twist_matrix_se3[(1,0)], 
                                twist_matrix_se3[(0,3)], twist_matrix_se3[(1,3)], twist_matrix_se3[(2,3)]];

    twist_r6
}



fn adjoint(
    // Computes the adjoint representation of a homoegenous transformation matrix
    // params: transform_bigse3 -- A homogeneous transformation matrix
    // returns: adjoint_transform -- The 6x6 adjoint representation of transform_bigse3 -- [Ad_transform_bigse3]
    transform_bigse3: &na::Matrix4<f64>,
) -> na::Matrix6<f64> {
    // form of adjoint representation is [Ad_transform_bigse3] = [ R, 0; [p]R, R ]
    //


    let (rotation_matrix, position_vector) = rotation_bigso3_position_r3_from_transform_bigse3(&transform_bigse3);

    let scew_position_rotation = scew_symmetric_matrix_so3_from_vector_r3(&position_vector) * rotation_matrix;

    let adjoint_transform = matrix![
        rotation_matrix[(0,0)]       , rotation_matrix[(0,1)]       , rotation_matrix[(0,2)]       , 0.0                   , 0.0                   , 0.0                   ;
        rotation_matrix[(1,0)]       , rotation_matrix[(1,1)]       , rotation_matrix[(1,2)]       , 0.0                   , 0.0                   , 0.0                   ;
        rotation_matrix[(2,0)]       , rotation_matrix[(2,1)]       , rotation_matrix[(2,2)]       , 0.0                   , 0.0                   , 0.0                   ;
        scew_position_rotation[(0,0)], scew_position_rotation[(0,1)], scew_position_rotation[(0,2)], rotation_matrix[(0,0)], rotation_matrix[(0,1)], rotation_matrix[(0,2)];
        scew_position_rotation[(1,0)], scew_position_rotation[(1,1)], scew_position_rotation[(1,2)], rotation_matrix[(1,0)], rotation_matrix[(1,1)], rotation_matrix[(1,2)];
        scew_position_rotation[(2,0)], scew_position_rotation[(2,1)], scew_position_rotation[(2,2)], rotation_matrix[(2,0)], rotation_matrix[(2,1)], rotation_matrix[(2,2)];
    ];

    adjoint_transform
}



fn normalized_screw_from_parametric_screw(
    // params: point_q -- A point lying on the screw axis, in vector form
    // params: vector_s -- A unit vector in the direction of the screw axid
    // params: pitch_h -- The pitch of the screw axis -- flesh out
    // returns: normalized_screw_axis -- A normalized 6-vector described by the input paramaters
    point_q: &na::Vector3<f64>,
    vector_s: &na::Vector3<f64>,
    pitch_h: &f64,
) -> na::Vector6<f64> {
    //
    let cross_prod_s_q = vector_s.cross(&point_q);
    let cross_prod_s_h = vector_s * *pitch_h;

    let linear_velocity = -cross_prod_s_q + cross_prod_s_h;

    let normalized_screw_axis = na::vector![
        vector_s[0], vector_s[1], vector_s[2],
        linear_velocity[0], linear_velocity[1], linear_velocity[2]
    ];

    normalized_screw_axis
}

/*
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form

    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S
 */


fn axis_angle_representation_r6_from_exponential_coordinates_r6(
    // Converts a 6-vector of exponential coordinates into screw axis-angle form
    // Essentially separates exponential coordinates into a normalized screw axis and its corresponding angle measurement
    // parmas: exponential_coordinates_r6 -- A 6-vector of exponential coordinates for rigid-body motion -- S*theta_angle
    // returns: screw_axis_r6 -- 6-vector normalized screw axis
    // returns: theta_angle -- The distance trabveled along or about screw_axis_r6
    exponential_coordinates_r6: &na::Vector6<f64>,
) -> (na::Vector6<f64>, f64) {
    //


    // theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    // first have theta_angle get the magnitude of exponential_coordinates_r6
    let first_three_exponential_coordinates_r6 = na::vector![
        exponential_coordinates_r6[0], exponential_coordinates_r6[1], exponential_coordinates_r6[2]
    ];
    let mut theta_angle = first_three_exponential_coordinates_r6.norm();
    // if theta_angle is near zero, redefine it using the linear velocity elements of exponential_coordinates_r6
    if near_zero(&theta_angle) {
        let last_three_exponential_coordinates_r6 = na::vector![
            exponential_coordinates_r6[3], exponential_coordinates_r6[4], exponential_coordinates_r6[5]
        ];
        theta_angle = last_three_exponential_coordinates_r6.norm();
    }

    let screw_axis_r6 = exponential_coordinates_r6 / theta_angle;

    (screw_axis_r6, theta_angle)
} 


/*

exponential_rotation_matrix_bigso3_from_exponential_coordinates_so3

 */


fn exponential_transformation_matrix_bigse3_from_exponential_coordinates_se3(
    // Computes the matrix exponential of an se3 representation of exponential coordinates
    // (ie if you have a 6-vector twist in its scew representation (a 4x4 matrix), this converts said matrix into a homogeneous transformation matrix that is the matrix exponential)
    // params: exponential_coordinate_matrix_se3 -- A 4x4 matrix representation of exponential coordinates
    // returns: exponential_transform_matrix_bigse3 -- A 4x4 homoegenous transformation matrix representing the matrix exponential of exponential_coordinate_matrix_se3
    exponential_coordinate_matrix_se3: &na::Matrix4<f64>,
) -> na::Matrix4<f64> {
    //

    let mut exponential_transform_matrix_bigse3 = na::matrix![
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;        
    ];

    let scew_exponential_coordinates = na::matrix![
        exponential_coordinate_matrix_se3[(0,0)], exponential_coordinate_matrix_se3[(0,1)], exponential_coordinate_matrix_se3[(0,2)];
        exponential_coordinate_matrix_se3[(1,0)], exponential_coordinate_matrix_se3[(1,1)], exponential_coordinate_matrix_se3[(1,2)];
        exponential_coordinate_matrix_se3[(2,0)], exponential_coordinate_matrix_se3[(2,1)], exponential_coordinate_matrix_se3[(2,2)];        
    ];


    let un_scewed_exponential_coordinates = vector_r3_from_scew_symmetric_matrix_so3(&scew_exponential_coordinates);

    if near_zero(&un_scewed_exponential_coordinates.norm()) {
        exponential_transform_matrix_bigse3 = na::matrix![
            1.0, 0.0, 0.0, exponential_coordinate_matrix_se3[(0,3)];
            0.0, 1.0, 0.0, exponential_coordinate_matrix_se3[(1,3)];
            0.0, 0.0, 1.0, exponential_coordinate_matrix_se3[(2,3)];
            0.0, 0.0, 0.0, 1.0;        
        ];
    } else {
        let (rotation_axis, theta_angle) = unit_vector_r3_angle_from_exponential_coordinates_r3(&un_scewed_exponential_coordinates);
        let normal_vector_r3 = check_for_normalization_r3(&rotation_axis);
        let scew_exponential_coordinates = scew_symmetric_matrix_so3_from_vector_r3(&normal_vector_r3);
        let scew_exponential_coordinates_squared = scew_exponential_coordinates * scew_exponential_coordinates;
        let rotation: na::Matrix3<f64> = na::Matrix3::identity() + theta_angle.sin() * scew_exponential_coordinates + (1.0 - theta_angle.cos()) * scew_exponential_coordinates_squared;

        let big_g_theta: na::Matrix3<f64> = na::Matrix3::identity() * theta_angle + (1.0 - theta_angle.cos()) * scew_exponential_coordinates + (theta_angle - theta_angle.sin()) * scew_exponential_coordinates_squared;

        let linear_velocity = na::vector![
            exponential_coordinate_matrix_se3[(0,3)], exponential_coordinate_matrix_se3[(1,3)], exponential_coordinate_matrix_se3[(2,3)]
        ];

        let big_g_velocity = big_g_theta * linear_velocity;

        exponential_transform_matrix_bigse3 = na::matrix![
            rotation[(0,0)], rotation[(0,0)], rotation[(0,0)], big_g_velocity[0];
            rotation[(0,0)], rotation[(0,0)], rotation[(0,0)], big_g_velocity[1];
            rotation[(0,0)], rotation[(0,0)], rotation[(0,0)], big_g_velocity[2];
            0.0, 0.0, 0.0, 1.0;        
        ];
    
    }


    exponential_transform_matrix_bigse3
}




fn _matrix_logarithm_se3_from_transformation_matrix_bigse3(
    //
) {
    //
}



// -- Chapter 3 -- //




// -- Chapter 4 -- //

// not implemented
// These two functions feel like they should be combined into a single function? Not sure
fn forward_kinematics_body(
    // Computes the forward kinemtaics for an open chain robot in the body frame (B)
    // params: home_configuration -- (M) -- The home configuration (position and orientation) of the end-effector
    // params: screw_axis_matrix_body_frame -- (Blist) -- The joint screw axes in the end-effector frame when the manipulator
    //                                                      is at the home position, in the format of a matrix with axes as the columns
    // params: joint_coordinate_list -- (thetalit) -- A list of joint coordinates
    // returns: forward_kinematics_transform_body -- A 4x4 homeogenous transformation matrix representing the position and orientation (configuration)
    //                                            of the end-effector of an open chain robot at a specific set of joint coordinates
    home_configuration: &na::Matrix4<f64>,
    // figure out declaration and use of dynamically sized matrices -- this might not work
    //screw_axis_matrix_body_frame: na::Matrix<f64, U6, Dyn, f64>,
    screw_axis_matrix_body_frame: &na::DMatrix<f64>,
    //joint_coordinate_list: na::Vector<f64, Dyn, f64>,
    joint_coordinate_list: &na::DVector<f64>,

) -> na::Matrix4<f64> {
    //

    // since screw_axis_matrix_body_frame is a dynamically sized matrix, we need to make sure it's the right size (some number of vectors of screw axes)

    let mut forward_kinematics_transform_body = matrix![
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
    ];

    //forward_kinematics_transform_body = home_configuration;

    for theta_angle in joint_coordinate_list.iter() {
        //forward_kinematics_transform_body = forward_kinematics_transform_body * 
    }

    forward_kinematics_transform_body
}


fn forward_kinematics_space(
    // Computes the forward kinemtaics for an open chain robot in the space frame (S)
) {
    //
}







// -- Chapter 4 -- //