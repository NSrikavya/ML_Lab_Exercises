from collections import Counter
def highest_occuring(inp_str):
  filter_char=[char.lower() for char in inp_str if char.isalpha()]
  if not filter_char:
    return "No alphabetic characters in input"
  char_count=Counter(filter_char)
  most_common,count=char_count.most_common(1)[0]
  return most_common,count
def main():
  user_input=input("Enter a string:")
  result=highest_occuring(user_input)
  if isinstance(result,str):
    print(result)
  else:
    char,count=result
    print(f"The highest occurring character is '{char}' with {count} occurrencse.")
main()