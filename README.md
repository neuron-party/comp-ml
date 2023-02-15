# computational machine learning lab

### notes to self:
* nohup.out is relay failure/generalization for ppo jumper agents
* nohup2.out is regular failure/generalization for ppo jumper agents
* nohup3.out is relay failure/generalization for ppo heist agents
* nohup4.out is regular failure/generalization for ppo heist agents
* a faster way to evaluate these metrics would be to terminate the environment after 500 steps instead of the default 1000 
* ppo_jumper_sta_2_final.pth # ppo agent trained for 200 million steps using upper bound STA
* ppo_jumper_sta3_2_final.pth # ppo agent trained for 200 million steps using more controllabe states and unbounded STA

* nohup50.out is same agent failure rates on jumper with the redfined successes/failures
* nohup51.out is the relay failure rates for jumper with the redefined successes/failures

### same agent vs relay failure rates
![same agent vs relay failures](docs/relay_metrics.png)
![training env returns](docs/train_test_returns_new.png)
![testing env returns](docs/train_test_returns_averaged_new.png)