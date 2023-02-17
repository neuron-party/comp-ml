# docs for metrics
`heist_relay_failures.pkl`: list of lists, where each inner list is 2000 relay evaluations (failures/successes) for heist <br>
`heist_sa_failures.pkl`: list of lists, where each inner list is 2000 same agent evaluations (failures/successes) for heist <br>
`jumper_relay_failures.pkl`: list of lists, where each inner list is 2000 relay evaluations for jumper <br>
`jumper_sa_failures.pkl`: list of lists, where each inner list is 2000 same agent evaluations <br>
`ppo_heist_and_jumper_train_test_rewards.pkl`: list of lists ordered as: [ppo_jumper_1_train_returns,..., ppo_jumper_4_train_returns, ppo_jumper_1_test_returns,..., ppo_jumper_4_test_returns, ppo_heist_1_train_returns, ..., ppo_heist_4_train_returns, ppo_heist_1_test_returns, ...., ppo_heist_4_test_returns], where each list of returns is from 100 runs. <br>
`ppo_ninja_and_leaper_train_test_rewards.pkl`: list of lists ordered as: [ppo_ninja_1_train_returns, ..., ppo_ninja_4_train_returns, ppo_ninja_1_test_returns, ..., ppo_ninja_4_test_returns, ppo_leaper_1_train_returns, ..., ppo_leaper_4_train_returns, ppo_leaper_1_test_returns, ..., ppo_leaper_4_test_returns], where each list of returns is from 100 runs. <br>
`jumper_relay_failures_redefined_final.pkl`: list of lists, where each inner list is 2000 relay evaluations for jumper. reran the initial experiments with the new definition of success (i.e finishing the trajectory between 400-500 steps is counted as a success) <br>
`jumper_sa_failures_redefined_final.pkl`: list of lists, where each inner list is 2000 same agent evaluations (with redefined successes) <br>
`ppo_jumper_sta_relay_failures_final.pkl`: list of lists, where each inner list is 2000 relay evaluations for the STA jumper agent. the main agent are the regular ppo agents, and the sta agent is always the relay runner. <br>
`ppo_jumper_sta_sa_failures_final.pkl`: list of 2000 same agent evaluations for the STA agent. <br>
`ppo_jumper_sta3_relay_failures_final.pkl`: list of lists, where each inner list is 2000 relay evaluations for the STA3 jumper agent. the main agents are the regular ppo agents, and the sta agent is always the relay runner. <br>
`ppo_jumper_sta3_sa_failures_final.pkl`: list of 2000 same agent evaluations for the STA3 agent. 