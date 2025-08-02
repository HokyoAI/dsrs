use crate::primatives::{Module, Signature};
use crate::providers::CompletionProvider;

struct Predict<S: Signature, P: CompletionProvider> {
    _marker: std::marker::PhantomData<S>,
    lm: P,
}

// impl<S: Signature, P: CompletionProvider> Module for Predict<S, P> {
//     type Sig = S;

//     fn aforward(
//         &self,
//         inputs: <<Self as Module>::Sig as Signature>::Inputs,
//     ) -> impl Future<Output = <<Self as Module>::Sig as Signature>::Outputs> {
//         self.lm.complete(messages, config)
//     }

//     fn parameters(&self) -> &[impl Module] {
//         let empty: &[impl Module] = &[];
//         empty
//     }
// }
