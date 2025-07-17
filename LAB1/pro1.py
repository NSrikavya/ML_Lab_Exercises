#program-1
def count_sum_pairs(arr,target):
  """
  count_sum_pairs() : function to count unique pairs in the array that equals 10
  arr : List of integers.
  target : The target sum to find pairs for.
  """
  visited=set() # Set to keep track of elements we have seen
  pairs= set() # Set to store unique pairs
  for num in arr: #  num-represents each individual element from arr
    diff=target-num # Calculate the difference of the current
    if diff in visited: # Check if the difference is in the set of visited elements
      pairs.add(tuple(sorted((num,diff))))
    visited.add(num) # Add the current element to the visited set
  return len(pairs) # Return the number of unique pairs

def main():
  arr=[2,7,4,1,3,6]
  target=10
  pairs=count_sum_pairs(arr,target)
  print("Number of pairs = ", pairs)

main()
