Part 1
---
- Produce a graph that shows the performance of DQN on the Cartpole domain with epsilon greedy exploration (annealed). This will serve as a baseline to compare DRQN. Run the given code without any modification and plot a graph where the x-axis is the number of episodes (2000 episodes) and the y-axis is the reward obtained. You can use a running average to smooth the plot if you so desire. Report curves that are the average of multiple runs of DQN (with different random seeds). 
  - (1 point) Produce a graph that shows the performance of the LSTM based DRQN on the Cartpole domain. Similar to the DQN case, the x-axis is the number of episodes (2000 episodes) and the y-axis is the reward obtained. You can use a running average to smooth the plot if you so desire. Report curves that are the average of multiple runs of DRQN (with different random seeds). 
![image](https://github.com/newyen2/1141_P75H400_HW3/blob/main/Part_1/1_DQN.png)
![image](https://github.com/newyen2/1141_P75H400_HW3/blob/main/Part_1/1_DRQN.png)

- (1 point) Based on the results explain the impact of the LSTM layer. Compare the performance of using DRQN and DQN. Elucidate the reasons for the observed difference in performance. 
  - 結果
    - DRQN的表現明顯好於DQN
    - DQN於600步時平穩，結果約為40，方差較小
    - DRQN於2000步時尚未平穩，結果約為150，方差較大
  - 分析
    - DQN無法有效處理POMDP環境
    - 其不具有時序記憶，因此無法辨別沒有速度訊息的環境
    - 而DRQN因為具有時序記憶，因此可以透過前後變化得到速度變化的訊息

Part 2
---
- Produce a graph that shows the performance of the given implementation of DQN on the Cartpole domain with epsilon greedy exploration (annealed). This will serve as a baseline to compare performance of the C51 algorithm. Run the given code without any modification and plot a graph where the x-axis is the number of episodes (at least 200 episodes) and the y-axis is the reward obtained per episode. You can use a running average to smooth the plot if you so desire. Report curves that are the average of multiple runs of the DQN (with different random seeds).
  - (1 point) Produce a graph that shows the performance of the C51 algorithm on the Cartpole domain. Similar to the DQN case, the x-axis is the number of episodes (500 episodes) and the y-axis is the reward obtained per episode. You can use a running average to smooth the plot if you so desire. Report curves that are the average of multiple runs of C51 (with different random seeds).
![image](https://github.com/newyen2/1141_P75H400_HW3/blob/main/Part_2/2_DQN.png)
![image](https://github.com/newyen2/1141_P75H400_HW3/blob/main/Part_2/2_C51.png)
  - (1 point) Based on the results, comment on the differences in the performance of C51 and DQN. What do the results suggest about the respective algorithms?
    - 結果
      - DQN於500步時平穩，結果約為400，方差較大
      - C51於500步時尚未平穩，結果約為450，方差較小
      - 兩者前期表現相似，但C51後期訓練結果較好
    - 分析
      - 由於CartPole屬於簡單環境，雖然題目有所改動，但這個程度的問題仍然是DQN能夠有效處理的環境
      - C51由於能夠對整個分布進行更新，因此在全局資訊上比DQN來得多，這可能是C51擁有更高分數上限的原因
