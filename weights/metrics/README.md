# docs for metrics
`heist_relay_failures.pkl`: list of lists, where each inner list is 2000 relay evaluations (failures/successes) for heist <br>
`heist_sa_failures.pkl`: list of lists, where each inner list is 2000 same agent evaluations (failures/successes) for heist <br>
`jumper_relay_failures.pkl`: list of lists, where each inner list is 2000 relay evaluations for jumper <br>
`jumper_sa_failures.pkl`: list of lists, where each inner list is 2000 same agent evaluations <br>
`ppo_heist_and_jumper_train_test_rewards.pkl`: list of lists ordered as: [ppo_jumper_1_train_returns,..., ppo_jumper_4_train_returns, ppo_jumper_1_test_returns,..., ppo_jumper_4_test_returns, ppo_heist_1_train_returns, ..., ppo_heist_4_train_returns, ..., ppo_heist_1_test_returns, ...., ppo_heist_4_test_returns], where each list of returns is from 100 runs.