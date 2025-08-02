use super::signature::Signature;
use std::future::Future;

pub trait Module {
    type Sig: Signature;

    fn forward(
        &self,
        inputs: <<Self as Module>::Sig as Signature>::Inputs,
    ) -> <<Self as Module>::Sig as Signature>::Outputs {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.aforward(inputs))
        })
    }

    fn aforward(
        &self,
        inputs: <<Self as Module>::Sig as Signature>::Inputs,
    ) -> impl Future<Output = <<Self as Module>::Sig as Signature>::Outputs>;

    fn parameters(&self) -> &[impl Module];
}
