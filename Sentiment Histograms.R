df = read.csv("/Users/lindegaard/Downloads/desc16_indicator.csv")
library(ggplot2)
df$X2020_1 = as.factor(df$X2020_1)
ggplot(df,aes(x=Sentiment.2016)) + 
  geom_histogram(data=subset(df,X2020_1 == 2016),aes(fill=X2020_1), alpha = 0.2) +
  geom_histogram(data=subset(df,X2020_1 == 2020),aes(fill=X2020_1), alpha = 0.2) +
  labs(x = "Lyric Sentiment Values", y = "Count", fill="Year") +
  scale_fill_manual(name="Year",values=c("blue","red")) +
  theme_minimal() 

ggplot(df,aes(x=Sentiment.2016)) + 
  geom_density(data=subset(df,X2020_1 == 2016),aes(fill=X2020_1), alpha = 0.7) +
  geom_density(data=subset(df,X2020_1 == 2020),aes(fill=X2020_1), alpha = 0.7) +
  labs(x = "Lyric Sentiment Values", y = "Count", fill="Year") +
  scale_fill_manual(name="Year",values = c("steelblue", "red")) +
  theme_minimal()
