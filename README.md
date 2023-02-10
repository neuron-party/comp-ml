# computational machine learning lab

### notes to self:
* nohup.out is relay failure/generalization for ppo jumper agents
* nohup2.out is regular failure/generalization for ppo jumper agents
* nohup3.out is relay failure/generalization for ppo heist agents
* nohup4.out is regular failure/generalization for ppo heist agents
* a faster way to evaluate these metrics would be to terminate the environment after 500 steps instead of the default 1000 

### same agent vs relay failure rates
![same agent vs relay failures](docs/relay_metrics.png)
![training env returns](docs/train_test_returns_new.png)
![testing env returns](docs/train_test_returns_averaged_new.png)