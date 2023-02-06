import json
import os
from tqdm import tqdm
from typing import List, TypedDict
from freebible import read_web
from langchain.embeddings import OpenAIEmbeddings

# load the bible
bible = read_web()

# get API key from top-right dropdown on OpenAI website
os.environ['OPENAI_API_KEY'] = 'Your API Key'

def get_books() -> List[str]:
    """Get list of books in bible."""
    return list(bible.keys())

# create a type for the verse groups
class VerseGroup(TypedDict):
    text: str # the text representation of the grouped verses 
    book: str # the book the verses are from
    chapter: str # the chapter the verses are from
    verses: str # the verses in the group
    embedding: List[float] | None # the embedding of the text

def split_text(book, verses_per_group: int) -> List[VerseGroup]:
    """Split bible into groups of specified verse count."""

    # create a list to store the grouped text
    verse_groups: List[VerseGroup] = []

    # loop through the chapters
    for chapter in range(1, len(bible[book]) + 1):
        # loop through the verses by groups
        for group in range(1, len(bible[book][chapter]), verses_per_group):
            # get the number of verses for the current group (last group may be smaller)
            verses_per_group = min(
                verses_per_group, len(bible[book][chapter]) - group)
            # create the group by appending the verses to a list
            context_window = [str(bible[book][chapter][group + i])
                              for i in range(verses_per_group)]
            # join the list into a string 
            current_group = "".join(context_window)
            # format the data
            verse_group: VerseGroup = { "text": current_group, "book": str(book), "chapter": str(chapter), "verses": f"{group}-{group + verses_per_group}", "embedding": None}
            # append the data to the list
            verse_groups.append(verse_group)

    return verse_groups

def get_embeddings(text_groups: List[VerseGroup]) -> List[VerseGroup]:
    # create the embeddings model
    model = OpenAIEmbeddings()

    # get the text from the text groups 
    texts = list(map(lambda group: group["text"], text_groups))
    
    # get the embeddings for the text
    embeddings = model.embed_documents(texts)

    # add the embeddings to the text groups
    for embedding, text_group in zip(embeddings, text_groups):
        text_group["embedding"] = embedding
    
    return text_groups

def save_embeddings_json(text_groups: List[VerseGroup], file_name: str):
    """Save the embedded verse groups to a json file."""
    with open(file_name, "w") as file:
        file.write(json.dumps(text_groups))


if __name__ == "__main__":

    # get the list of books
    books = get_books()
    # set the number of verses per group
    verse_group_sizes = [1, 6, 12]
    # loop through the books
    for book in tqdm(books):
        # loop through the verse group sizes
        for verse_group_size in verse_group_sizes:
            # split the text into groups
            verse_groups = split_text(book, verse_group_size)
            # get the embeddings
            verse_groups = get_embeddings(verse_groups)
            # save the embeddings to a json file
            save_embeddings_json(verse_groups, f"{book}_{verse_group_size}.json")
            print(f"Saved {book} {verse_group_size} verses per group")
        
        
