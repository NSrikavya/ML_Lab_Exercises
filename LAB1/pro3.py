import numpy as np
def is_square_matrix(matrix):#function to find if the matrix is a square matrix or not
  return all(len(row) == len(matrix) for row in matrix)
def matrix_power(matrix,power):#fumction to compute A^m for a square matrix
  if not is_square_matrix(matrix):
    return "Error:Input must be a squre matrix."
  if power <= 0:
    return "Error:Power must be positive"
  np_matrix=np.array(matrix)
  result = np.linalg.matrix_power(np_matrix,power)
  return result
def matrix_input():
  n=int(input("Enter the size of the matrix:"))
  matrix=[]
  for i in range(n):
    row = list(map(int,input(f"Row{i+1}:").strip().split()))
    if len(row) != n:
      return None
    matrix.append(row)
  return matrix
def main():
  matrix=matrix_input()
  if matrix is None:
    return
  try:
    power=int(input("Enter the position integer power m:"))
  except ValueError:
    print("Error:Power must be an integer.")
    return
  result=matrix_power(matrix,power)
  print("A^m is:")
  print(result)
main()
