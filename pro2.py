def calc_range(arr):
  #arr-array of elements
  if len(arr) < 3:
    return "Range determination  not possible"
  maximum=max(arr)#finds maximum of the array
  minimum=min(arr)#finds minimum of the array
  range_num=maximum-minimum#finds the range of the array
  return f"The range is {range_num}"#return the range found

def main():
  num=[5,3,8,1,0,4]
  result=calc_range(num)
  print(result)
main()