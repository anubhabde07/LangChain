from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
def lcs_length(s1, s2):
    memo = {}

    def solve(i, j):
        # Base case: if either string is exhausted
        if i == len(s1) or j == len(s2):
            return 0
        
        # Check if result is already in memo
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Recursive step: 
        if s1[i] == s2[j]:
            # If characters match, add 1 and recurse on the rest of the strings
            result = 1 + solve(i + 1, j + 1)
        else:
            # If characters don't match, take the maximum of skipping a character 
            # from either string
            result = max(solve(i + 1, j), solve(i, j + 1))
        
        # Store result in memo and return
        memo[(i, j)] = result
        return result

    return solve(0, 0)

# Example Usage:
string1 = "AGGTAB"
string2 = "GXTXAYB"

length = lcs_length(string1, string2)
print(f"String 1: {string1}")
print(f"String 2: {string2}")
print(f"Length of LCS: {length}") 
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])