# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tune momentum as well as learning rate using local quadratic approximation.

The implementation here can be **a drop-in replacement for any optimizer in MXNet** (So far we only implemented and tested upon SGD and other optimizers are in the to-do list).

For more technical details, please refer to the paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471).
