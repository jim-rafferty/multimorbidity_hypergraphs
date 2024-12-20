
use ndarray::Axis;
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

pub fn degree_centrality(
    h: &HypergraphBase, 
    rep: Representation,
    weighted: bool
) -> Vec<f64> {
    
    match (rep, weighted) {
        (Representation::Standard, true) => {
            let inc_mat: CsMat<_> = CsMat::csr_from_dense(
                h.incidence_matrix.mapv(|x| x as f64).view(), 0.0
            );
            let w = diag_sprs(&h.edge_weights);
            let m = &w * & inc_mat;
            m.to_dense().sum_axis(Axis(0)).to_vec()
        }
        (Representation::Standard, false) => {
            h.incidence_matrix
                .mapv(|x| x as f64)
                .sum_axis(Axis(0))
                .to_vec()
        }
        (Representation::Dual, true) => {
            let inc_mat: CsMat<_> = CsMat::csr_from_dense(
                h.incidence_matrix.mapv(|x| x as f64).view(), 0.0
            );
            let w = diag_sprs(&h.node_weights);
            let m = &inc_mat * &w;
            m.to_dense().sum_axis(Axis(1)).to_vec()
        }
        (Representation::Dual, false) => {
            h.incidence_matrix
                .mapv(|x| x as f64)
                .sum_axis(Axis(1))
                .to_vec()
        }
        (_, _) => panic!("This set of optional arguments not supported"),
    }
}