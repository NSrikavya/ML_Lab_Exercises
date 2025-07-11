import random
import statistics as st
def generate_list(n,lower,upper):
  return [random.randint(lower,upper) for _ in range(n)]
def compute_stats(num):
  mean_val=st.mean(num)
  median_val=st.median(num)
  mode_val=st.mode(num)
  return mean_val,median_val,mode_val
def main():
  random_lis=generate_list(25,1,10)
  mean,median,mode = compute_stats(random_lis)
  print("Mean:",mean)
  print("Median:",median)
  print("Mode:",mode)
main()