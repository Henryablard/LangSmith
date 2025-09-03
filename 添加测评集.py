from langsmith import Client

client = Client()

dataset_name = "Sample dataset V2"

# 新样例
examples = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris."},
    },
    {
        "inputs": {"question": "Who wrote the play 'Romeo and Juliet'?"},
        "outputs": {"answer": "William Shakespeare."},
    },
    {
        "inputs": {"question": "What is the chemical symbol for water?"},
        "outputs": {"answer": "H2O."},
    },
    {
        "inputs": {"question": "Which planet is known as the Red Planet?"},
        "outputs": {"answer": "Mars."},
    },
    {
        "inputs": {"question": "Who painted the Mona Lisa?"},
        "outputs": {"answer": "Leonardo da Vinci."},
    },
    {
        "inputs": {"question": "What is the largest mammal in the world?"},
        "outputs": {"answer": "The blue whale."},
    },
    {
        "inputs": {"question": "In which year did World War II end?"},
        "outputs": {"answer": "1945."},
    },
    {
        "inputs": {"question": "What is the freezing point of water in Celsius?"},
        "outputs": {"answer": "0 degrees Celsius."},
    },
    {
        "inputs": {"question": "Which country is famous for the Great Wall?"},
        "outputs": {"answer": "China."},
    },
    {
        "inputs": {"question": "Who was the first person to walk on the moon?"},
        "outputs": {"answer": "Neil Armstrong."},
    },
    {
        "inputs": {"question": "What is the smallest prime number?"},
        "outputs": {"answer": "2."},
    },
    {
        "inputs": {"question": "Which gas do humans need to breathe in order to survive?"},
        "outputs": {"answer": "Oxygen."},
    },
    {
        "inputs": {"question": "What is the tallest mountain in the world?"},
        "outputs": {"answer": "Mount Everest."},
    },
    {
        "inputs": {"question": "Which ocean is the largest on Earth?"},
        "outputs": {"answer": "The Pacific Ocean."},
    },
    {
        "inputs": {"question": "Who developed the theory of general relativity?"},
        "outputs": {"answer": "Albert Einstein."},
    },
    {
        "inputs": {"question": "What is the hardest natural substance on Earth?"},
        "outputs": {"answer": "Diamond."},
    },
    {
        "inputs": {"question": "Which country gifted the Statue of Liberty to the United States?"},
        "outputs": {"answer": "France."},
    },
    {
        "inputs": {"question": "What is the boiling point of water in Celsius?"},
        "outputs": {"answer": "100 degrees Celsius."},
    },
    {
        "inputs": {"question": "Who is known as the Father of Computers?"},
        "outputs": {"answer": "Charles Babbage."},
    },
    {
        "inputs": {"question": "Which desert is the largest in the world?"},
        "outputs": {"answer": "The Sahara Desert."},
    },
]


# 批量添加
client.create_examples(dataset_name=dataset_name, examples=examples)
