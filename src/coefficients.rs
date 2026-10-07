//! Method coefficients for the Adams and BDF methods (`DCFODE`).
//!
//! The tables mirror the Fortran arrays `ELCO(13,12)` and `TESCO(3,12)`:
//! `elco[nq - 1][i - 1]` is `ELCO(i, nq)` and `tesco[nq - 1][k - 1]` is
//! `TESCO(k, nq)`.
//!
//! * `ELCO(i, nq)`, `1 <= i <= nq + 1`, are the coefficients of the
//!   generating polynomial `l(x) = el(1) + el(2) x + ... + el(nq+1) x^nq` of
//!   the method of order `nq`. For Adams, `dl/dx = (x+1)(x+2)...(x+nq-1) /
//!   (nq-1)!` with `l(-1) = 0`; for BDF, `l(x) = (x+1)(x+2)...(x+nq) / K` with
//!   `K = nq! (1 + 1/2 + ... + 1/nq)`.
//! * `TESCO(k, nq)` are the test constants for the step-size selection at
//!   order `nq - 1` (`k = 1`), `nq` (`k = 2`) and `nq + 1` (`k = 3`).

/// Integration method (`METH`): 1 = nonstiff Adams, 2 = stiff BDF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    /// Implicit Adams, orders 1 to 12 (`METH = 1`).
    Adams,
    /// Backward differentiation formulas, orders 1 to 5 (`METH = 2`).
    Bdf,
}

impl Method {
    /// The Fortran `METH` code (1 or 2), also the `MUSED`/`MCUR` value.
    pub fn code(self) -> i32 {
        match self {
            Self::Adams => 1,
            Self::Bdf => 2,
        }
    }

    /// Largest order DCFODE fills for this method (12 or 5).
    pub fn max_order(self) -> usize {
        match self {
            Self::Adams => crate::consts::MORD_NONSTIFF,
            Self::Bdf => crate::consts::MORD_STIFF,
        }
    }
}

/// `ELCO(13,12)` storage: `elco[nq - 1][i - 1]` = `ELCO(i, nq)`.
pub type Elco = [[f64; 13]; 12];
/// `TESCO(3,12)` storage: `tesco[nq - 1][k - 1]` = `TESCO(k, nq)`.
pub type Tesco = [[f64; 3]; 12];

/// Fills `elco` and `tesco` for `meth` (`DCFODE`).
///
/// Like the Fortran routine this only writes the entries the method defines
/// (orders `1..=12` for Adams, `1..=5` for BDF); everything else in the
/// arrays is left untouched. DSTODA keeps these arrays across method
/// switches, so the in-place semantics are preserved deliberately.
pub fn cfode(meth: Method, elco: &mut Elco, tesco: &mut Tesco) {
    // `pc[i - 1]` is `PC(i)`.
    let mut pc = [0.0_f64; 12];
    match meth {
        Method::Adams => {
            elco[0][0] = 1.0;
            elco[0][1] = 1.0;
            tesco[0][0] = 0.0;
            tesco[0][1] = 2.0;
            tesco[1][0] = 1.0;
            tesco[11][2] = 0.0;
            pc[0] = 1.0;
            let mut rqfac = 1.0_f64;
            for nq in 2..=12usize {
                // PC holds the coefficients of p(x) = (x+1)(x+2)...(x+nq-1).
                let rq1fac = rqfac;
                rqfac /= nq as f64;
                let nqm1 = nq - 1;
                let fnqm1 = nqm1 as f64;
                let nqp1 = nq + 1;
                // Form coefficients of p(x)*(x+nq-1).
                pc[nq - 1] = 0.0;
                for ib in 1..=nqm1 {
                    let i = nqp1 - ib;
                    pc[i - 1] = pc[i - 2] + fnqm1 * pc[i - 1];
                }
                pc[0] *= fnqm1;
                // Compute integral, -1 to 0, of p(x) and x*p(x).
                let mut pint = pc[0];
                let mut xpin = pc[0] / 2.0;
                let mut tsign = 1.0_f64;
                for i in 2..=nq {
                    tsign = -tsign;
                    pint += tsign * pc[i - 1] / i as f64;
                    xpin += tsign * pc[i - 1] / (i + 1) as f64;
                }
                // Store coefficients in ELCO and TESCO.
                elco[nq - 1][0] = pint * rq1fac;
                elco[nq - 1][1] = 1.0;
                for i in 2..=nq {
                    elco[nq - 1][i] = rq1fac * pc[i - 1] / i as f64;
                }
                let agamq = rqfac * xpin;
                let ragq = 1.0 / agamq;
                tesco[nq - 1][1] = ragq;
                if nq < 12 {
                    tesco[nqp1 - 1][0] = ragq * rqfac / nqp1 as f64;
                }
                tesco[nqm1 - 1][2] = ragq;
            }
        }
        Method::Bdf => {
            pc[0] = 1.0;
            let mut rq1fac = 1.0_f64;
            for nq in 1..=5usize {
                // PC holds the coefficients of p(x) = (x+1)(x+2)...(x+nq).
                let fnq = nq as f64;
                let nqp1 = nq + 1;
                // Form coefficients of p(x)*(x+nq).
                pc[nqp1 - 1] = 0.0;
                for ib in 1..=nq {
                    let i = nq + 2 - ib;
                    pc[i - 1] = pc[i - 2] + fnq * pc[i - 1];
                }
                pc[0] *= fnq;
                // Store coefficients in ELCO and TESCO.
                for i in 1..=nqp1 {
                    elco[nq - 1][i - 1] = pc[i - 1] / pc[1];
                }
                elco[nq - 1][1] = 1.0;
                tesco[nq - 1][0] = rq1fac;
                tesco[nq - 1][1] = nqp1 as f64 / elco[nq - 1][0];
                tesco[nq - 1][2] = (nq + 2) as f64 / elco[nq - 1][0];
                rq1fac /= fnq;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact rational arithmetic for an independent reference computation.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Q(i128, i128);

    fn gcd(a: i128, b: i128) -> i128 {
        if b == 0 { a.abs() } else { gcd(b, a % b) }
    }

    impl Q {
        fn new(n: i128, d: i128) -> Q {
            assert!(d != 0);
            let g = gcd(n, d).max(1) * d.signum();
            Q(n / g, d / g)
        }
        fn int(n: i128) -> Q {
            Q(n, 1)
        }
        fn add(self, o: Q) -> Q {
            Q::new(self.0 * o.1 + o.0 * self.1, self.1 * o.1)
        }
        fn mul(self, o: Q) -> Q {
            Q::new(self.0 * o.0, self.1 * o.1)
        }
        fn div(self, o: Q) -> Q {
            Q::new(self.0 * o.1, self.1 * o.0)
        }
        fn f(self) -> f64 {
            self.0 as f64 / self.1 as f64
        }
    }

    /// Coefficients (ascending powers) of prod_{k=1..m} (x + k).
    fn rising(m: usize) -> Vec<Q> {
        let mut p = vec![Q::int(1)];
        for k in 1..=m as i128 {
            let mut q = vec![Q::int(0); p.len() + 1];
            for (i, c) in p.iter().enumerate() {
                q[i] = q[i].add(c.mul(Q::int(k)));
                q[i + 1] = q[i + 1].add(*c);
            }
            p = q;
        }
        p
    }

    fn fact(n: usize) -> i128 {
        (1..=n as i128).product()
    }

    /// Integral over [-1, 0] of the polynomial with coefficients `p`.
    fn integral_m1_0(p: &[Q]) -> Q {
        // int_{-1}^{0} x^i dx = (-1)^i / (i + 1).
        p.iter().enumerate().fold(Q::int(0), |acc, (i, c)| {
            let s = if i % 2 == 0 { 1 } else { -1 };
            acc.add(c.mul(Q::new(s, i as i128 + 1)))
        })
    }

    /// Exact Adams `el` for order `nq` from `dl/dx = rising(nq-1)/(nq-1)!`,
    /// `l(-1) = 0`.
    fn adams_el(nq: usize) -> Vec<Q> {
        let p = rising(nq - 1);
        let rfac = Q::new(1, fact(nq - 1));
        let mut el = vec![Q::int(0); nq + 1];
        for (i, c) in p.iter().enumerate() {
            el[i + 1] = c.mul(rfac).div(Q::int(i as i128 + 1));
        }
        // l(-1) = 0 => el(1) = int_{-1}^{0} dl/dx dx.
        el[0] = integral_m1_0(&p).mul(rfac);
        el
    }

    /// Exact Adams test constants as defined in DCFODE:
    /// `TESCO(2,nq) = 1/gamma(nq)` with `gamma(nq) = -int_{-1}^0 x p(x) dx / nq!`,
    /// `TESCO(1,nq+1) = TESCO(2,nq) / (nq! (nq+1))`, `TESCO(3,nq-1) = TESCO(2,nq)`.
    fn adams_ragq(nq: usize) -> Q {
        let p = rising(nq - 1);
        let mut xp = vec![Q::int(0)];
        xp.extend_from_slice(&p);
        let xpin = Q::int(0).add(integral_m1_0(&xp).mul(Q::int(-1)));
        Q::int(1).div(xpin.div(Q::int(fact(nq))))
    }

    fn bdf_el(nq: usize) -> Vec<Q> {
        let p = rising(nq);
        p.iter().map(|c| c.div(p[1])).collect()
    }

    fn assert_close(got: f64, want: f64, what: &str) {
        let tol = 4.0 * f64::EPSILON * want.abs();
        assert!((got - want).abs() <= tol, "{what}: got {got:e}, want {want:e}");
    }

    fn tables(meth: Method) -> (Elco, Tesco) {
        let mut elco = [[0.0; 13]; 12];
        let mut tesco = [[0.0; 3]; 12];
        cfode(meth, &mut elco, &mut tesco);
        (elco, tesco)
    }

    #[test]
    fn coefficients_bdf_transcribed_table() {
        // Classic BDF coefficients (Hindmarsh, LSODE report, Table 2).
        let el: [&[f64]; 5] = [
            &[1.0, 1.0],
            &[2.0 / 3.0, 1.0, 1.0 / 3.0],
            &[6.0 / 11.0, 1.0, 6.0 / 11.0, 1.0 / 11.0],
            &[12.0 / 25.0, 1.0, 7.0 / 10.0, 1.0 / 5.0, 1.0 / 50.0],
            &[60.0 / 137.0, 1.0, 225.0 / 274.0, 85.0 / 274.0, 15.0 / 274.0, 1.0 / 274.0],
        ];
        // TESCO(1,nq) = 1/(nq-1)!, TESCO(2,nq) = (nq+1)/el0, TESCO(3,nq) = (nq+2)/el0.
        let tesco_want: [[f64; 3]; 5] = [
            [1.0, 2.0, 3.0],
            [1.0, 4.5, 6.0],
            [0.5, 22.0 / 3.0, 55.0 / 6.0],
            [1.0 / 6.0, 125.0 / 12.0, 12.5],
            [1.0 / 24.0, 13.7, 959.0 / 60.0],
        ];
        let (elco, tesco) = tables(Method::Bdf);
        for nq in 1..=5 {
            for (i, &w) in el[nq - 1].iter().enumerate() {
                assert_close(elco[nq - 1][i], w, &format!("BDF ELCO({},{nq})", i + 1));
            }
            for k in 0..3 {
                assert_close(tesco[nq - 1][k], tesco_want[nq - 1][k], &format!("BDF TESCO({},{nq})", k + 1));
            }
        }
    }

    #[test]
    fn coefficients_bdf_match_exact_definition() {
        let (elco, _) = tables(Method::Bdf);
        for nq in 1..=5 {
            for (i, q) in bdf_el(nq).iter().enumerate() {
                assert_close(elco[nq - 1][i], q.f(), &format!("BDF ELCO({},{nq})", i + 1));
            }
        }
    }

    #[test]
    fn coefficients_adams_transcribed_table() {
        let el: [&[f64]; 6] = [
            &[1.0, 1.0],
            &[1.0 / 2.0, 1.0, 1.0 / 2.0],
            &[5.0 / 12.0, 1.0, 3.0 / 4.0, 1.0 / 6.0],
            &[3.0 / 8.0, 1.0, 11.0 / 12.0, 1.0 / 3.0, 1.0 / 24.0],
            &[251.0 / 720.0, 1.0, 25.0 / 24.0, 35.0 / 72.0, 5.0 / 48.0, 1.0 / 120.0],
            &[95.0 / 288.0, 1.0, 137.0 / 120.0, 5.0 / 8.0, 17.0 / 96.0, 1.0 / 40.0, 1.0 / 720.0],
        ];
        let (elco, tesco) = tables(Method::Adams);
        for nq in 1..=6 {
            for (i, &w) in el[nq - 1].iter().enumerate() {
                assert_close(elco[nq - 1][i], w, &format!("Adams ELCO({},{nq})", i + 1));
            }
        }
        // Constants set explicitly by DCFODE ...
        assert_eq!(tesco[0][0], 0.0);
        assert_eq!(tesco[0][1], 2.0);
        assert_eq!(tesco[1][0], 1.0);
        assert_eq!(tesco[11][2], 0.0);
        // ... and the first computed ones.
        let want: [[f64; 3]; 2] = [[0.0, 2.0, 12.0], [1.0, 12.0, 24.0]];
        for nq in 1..=2 {
            for k in 0..3 {
                assert_close(tesco[nq - 1][k], want[nq - 1][k], &format!("Adams TESCO({},{nq})", k + 1));
            }
        }
    }

    #[test]
    fn coefficients_adams_match_exact_definition() {
        let (elco, tesco) = tables(Method::Adams);
        for nq in 1..=12 {
            let el = adams_el(nq);
            for (i, q) in el.iter().enumerate() {
                assert_close(elco[nq - 1][i], q.f(), &format!("Adams ELCO({},{nq})", i + 1));
            }
        }
        for nq in 2..=12 {
            let ragq = adams_ragq(nq);
            assert_close(tesco[nq - 1][1], ragq.f(), &format!("Adams TESCO(2,{nq})"));
            assert_close(tesco[nq - 2][2], ragq.f(), &format!("Adams TESCO(3,{})", nq - 1));
            if nq < 12 {
                let t1 = ragq.div(Q::int(fact(nq))).div(Q::int(nq as i128 + 1));
                assert_close(tesco[nq][0], t1.f(), &format!("Adams TESCO(1,{})", nq + 1));
            }
        }
    }

    #[test]
    fn coefficients_cfode_leaves_unused_entries_untouched() {
        // Fill with Adams, then switch to BDF as DSTODA does: orders 6..12
        // and ELCO(i, nq) for i > nq + 1 keep their previous values.
        let (mut elco, mut tesco) = tables(Method::Adams);
        let (adams_elco, adams_tesco) = (elco, tesco);
        cfode(Method::Bdf, &mut elco, &mut tesco);
        assert_eq!(elco[5..], adams_elco[5..]);
        assert_eq!(tesco[5..], adams_tesco[5..]);
        assert_eq!(elco[0][2..], adams_elco[0][2..]);
        assert_eq!(elco[4][6..], adams_elco[4][6..]);
    }

    #[test]
    fn coefficients_method_codes() {
        assert_eq!(Method::Adams.code(), 1);
        assert_eq!(Method::Bdf.code(), 2);
        assert_eq!(Method::Adams.max_order(), 12);
        assert_eq!(Method::Bdf.max_order(), 5);
    }
}
