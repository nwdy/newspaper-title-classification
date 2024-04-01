from bs4 import BeautifulSoup
import csv


def extract_data(filename):
    file_path = 'data-mining/Project/Crawler/content-web/' + filename + 'l'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except IOError as e:
        print("An error occurred while reading the file:", e)

    soup = BeautifulSoup(content, 'html.parser')
    a_tags = soup.select('div.list__stream-flex div.box-category-middle a.box-category-link-title')
    
    index = 0
    result = []

    genre = filename[:-4]

    for a_tag in a_tags:
        if index < 250:
            result.append([a_tag.get('title'), genre])
            index += 1
        else:
            break
    
    return result
        
def write_data(result):
    with open('data-mining/Project/Crawler/data/newspaper.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(['id', 'title', 'genre'])

        for idx, row in enumerate(result, start=1):
            writer.writerow([idx] + row)


def main():
    filenames = [
                 'kinh-te.htm', 
                 'giao-duc.htm', 
                 'xe.htm', 
                 'suc-khoe.htm', 
                 'cong-nghe-game.htm'
                 ]
    
    result = []

    for filename in filenames:
        result += extract_data(filename)
    
    write_data(result)


if __name__ == '__main__':
    main()