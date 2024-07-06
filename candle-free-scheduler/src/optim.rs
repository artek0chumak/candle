use candle::{Result, Var};
use candle_nn::Optimizer;

#[derive(Clone, Debug)]
pub struct ParamsSGDMomentum {
    pub lr: f64,
    pub beta: f64,
}

impl Default for ParamsSGDMomentum {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta: 0.1,
        }
    }
}

#[derive(Debug)]
struct VarSGDMomentum {
    var: Var,
    first_moment: Var,
}

#[derive(Debug)]
pub struct SGDMomentum {
    vars: Vec<VarSGDMomentum>,
    step_t: usize,
    params: ParamsSGDMomentum,
}

impl Optimizer for SGDMomentum {
    type Config = ParamsSGDMomentum;

    fn new(vars: Vec<Var>, params: ParamsSGDMomentum) ->  Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarSGDMomentum {
                    var,
                    first_moment,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> candle::Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let beta = self.params.beta;
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta)? + g)?;
                let next_theta = (theta.as_tensor() - (&next_m * lr)?)?;
                m.set(&next_m)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl SGDMomentum {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> candle::Result<Self> {
        let params = ParamsSGDMomentum {
            lr: learning_rate,
            ..ParamsSGDMomentum::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsSGDMomentum {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsSGDMomentum) {
        self.params = params;
    }
}

#[derive(Clone, Debug)]
pub struct ParamsSGDSchedulerFree {
    pub lr: f64,
    pub beta: f64,
}

impl Default for crate::optim::ParamsSGDSchedulerFree {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta: 0.9,
        }
    }
}

#[derive(Debug)]
struct VarSGDSchedulerFree {
    var: Var,
    x: Var,
    z: Var,
}

#[derive(Debug)]
pub struct SGDSchedulerFree {
    vars: Vec<crate::optim::VarSGDSchedulerFree>,
    step_t: usize,
    params: ParamsSGDSchedulerFree,
}

impl Optimizer for SGDSchedulerFree {
    type Config = ParamsSGDSchedulerFree;

    fn new(vars: Vec<Var>, params: ParamsSGDSchedulerFree) ->  Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let x = Var::zeros(shape, dtype, device)?;
                let z = Var::zeros(shape, dtype, device)?;
                Ok(VarSGDSchedulerFree {
                    var,
                    x,
                    z,
                })
            })
            .collect::<candle::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> candle::Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let beta = self.params.beta;
        let c = 1.0 / (self.step_t as f64);
        for var in self.vars.iter() {
            let y = &var.var;
            let x = &var.x;
            let z = &var.z;
            if let Some(g) = grads.get(y) {
                let next_y = (((1.0 - beta) * z.as_tensor())? + beta * x.as_tensor())?;
                y.set(&next_y)?;
                let next_z = (z.as_tensor() - lr * g)?;
                let next_x  = (((1.0 - c) * x.as_tensor())? + c * &next_z)?;
                z.set(&next_z)?;
                x.set(&next_x)?;
            }
        }
        Ok(())
    }
}

impl SGDSchedulerFree {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> candle::Result<Self> {
        let params = ParamsSGDSchedulerFree {
            lr: learning_rate,
            ..ParamsSGDSchedulerFree::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsSGDSchedulerFree {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsSGDSchedulerFree) {
        self.params = params;
    }
}
