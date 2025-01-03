
use ndarray::{Axis, Array1, Array2};
use sprs::{CsMat, TriMat};

use crate::types::*;

pub fn diag_sprs(
    v: &Vec<f64>,
) -> CsMat<f64> {
    
    let mut a = TriMat::new((v.len(), v.len()));
    
    for (i, x) in v.iter().enumerate() {
        a.add_triplet(i, i, *x);
    }
    a.to_csr()
}

pub fn degree_centrality<T: std::clone::Clone>(
    incidence_matrix: &Array2<T>, 
    rep: Representation,
    weights: Option<Vec<f64>> 
) -> Vec<f64> where f64: From<T>{
    
    match (rep, weights) {
        (Representation::Standard, Some(weight_array)) => {
            let inc_mat: CsMat<_> = CsMat::csr_from_dense(
                incidence_matrix.mapv(|x| f64::from(x)).view(), 0.0
            );
            let w = diag_sprs(&weight_array);
            let m = &w * &inc_mat;
            m.to_dense().sum_axis(Axis(0)).to_vec()
        }
        (Representation::Standard, None) => {
            incidence_matrix
                .mapv(|x| f64::from(x))
                .sum_axis(Axis(0))
                .to_vec()
        }
        (Representation::Dual, Some(weight_array)) => {
            let inc_mat: CsMat<_> = CsMat::csr_from_dense(
                incidence_matrix.mapv(|x| f64::from(x)).view(), 0.0
            );
            let w = diag_sprs(&weight_array);
            let m = &inc_mat * &w;
            m.to_dense().sum_axis(Axis(1)).to_vec()
        }
        (Representation::Dual, None) => {
            incidence_matrix
                .mapv(|x| f64::from(x))
                .sum_axis(Axis(1))
                .to_vec()
        }
        (_, _) => panic!("This set of optional arguments not supported"),
    }
}