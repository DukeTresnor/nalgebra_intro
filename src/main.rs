extern crate nalgebra as na;

use na::matrix;
//use na::Unit;


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

    let exp_coor_test = exponential_rotation_matrix_bigso3_from_exponential_coordinates_r3(&logarithm_output_for_logarithm_test);

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
) -> (na::Vector3<f64>, f64){
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
fn exponential_rotation_matrix_bigso3_from_exponential_coordinates_r3(
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

    let dummy_return = matrix![
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
    ];

    let un_scewed_exponential_coordinates = vector_r3_from_scew_symmetric_matrix_so3(skew_exponential_coordinates);

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



// -- Chapter 3 -- //
