"""Make a zip file for submission."""

import os
import pandas as pd

SRC_EXT = '.xlsx'

def make_datafiles():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pages = []
    isbns = []
    levels = []
    titles = []
    wc = []
    batch = []
    page_nums = []
    print('Walking files...')
    book_list_df = pd.read_csv('../cs229_sp22_dataset/book_list.csv')
    isbn_title_dictionary = dict(zip(book_list_df.ISBN, book_list_df.Title))
    isbn_level_dictionary = dict(zip(book_list_df.ISBN, book_list_df.Level))
    all_isbn_strings = [str(s) for s in isbn_level_dictionary.keys()]
    for base_path, dir_names, file_names in os.walk('../cs229_sp22_dataset'):
        for file_name in file_names:
            if file_name.endswith(SRC_EXT):
                # Read file
                file_path = os.path.join(base_path, file_name)
                rel_path = os.path.relpath(file_path, script_dir)
                batch_name = None
                page_loc = None
                if 'batch1' in rel_path:
                    df = pd.read_excel(rel_path, header=None)
                    curr_pages = df[1]
                    begin = rel_path.rfind('/')
                    end = rel_path.rfind('.')
                    isbn =  rel_path[begin + 1:end]
                    if isbn in all_isbn_strings:
                        page = 1
                        for pg in curr_pages:
                            if pg != pg:
                                pg = ""
                            isbns.append(isbn)
                            pages.append(pg)
                            wc.append(len(pg))
                            levels.append(isbn_level_dictionary[int(isbn)])
                            titles.append(isbn_title_dictionary[int(isbn)])
                            batch.append('batch1')
                            page_nums.append(page)
                            page += 1
                if 'batch2' in rel_path:
                    df = pd.read_excel(rel_path, header=None)
                    curr_pages = df[0]
                    begin = rel_path.rfind('/')
                    end = rel_path.rfind('.')
                    isbn =  rel_path[begin + 1:end]
                    if isbn in all_isbn_strings:
                        page = 1
                        for pg in curr_pages:
                            if pg != pg:
                                pg = ""
                            isbns.append(isbn)
                            pages.append(pg)
                            wc.append(len(pg))
                            levels.append(isbn_level_dictionary[int(isbn)])
                            titles.append(isbn_title_dictionary[int(isbn)])
                            batch.append('batch2')
                            page_nums.append(page)
                            page += 1

                if 'batch3' in rel_path:
                    df = pd.read_excel(rel_path, header=0)
                    curr_pages = df['Text']
                    begin = rel_path.rfind('/')
                    end = rel_path.rfind('.')
                    isbn =  rel_path[begin + 1:end]
                    if isbn in all_isbn_strings:
                        page = 1
                        for pg in curr_pages:
                            if pg != pg:
                                pg = ""
                            isbns.append(isbn)
                            pages.append(pg)
                            wc.append(len(pg))
                            levels.append(isbn_level_dictionary[int(isbn)])
                            titles.append(isbn_title_dictionary[int(isbn)])
                            batch.append('batch3')
                            page_nums.append(page)
                            page += 1


                if 'batch4' in rel_path:
                    df = pd.read_excel(rel_path, header=0)
                    curr_pages = df['Text']
                    begin = rel_path.rfind('/')
                    end = rel_path.rfind('.')
                    isbn =  rel_path[begin + 1:end]
                    if isbn in all_isbn_strings:
                        page = 1
                        for pg in curr_pages:
                            if pg != pg:
                                pg = ""
                            isbns.append(isbn)
                            pages.append(pg)
                            wc.append(len(pg))
                            levels.append(isbn_level_dictionary[int(isbn)])
                            titles.append(isbn_title_dictionary[int(isbn)])
                            batch.append('batch4')
                            page_nums.append(page)
                            page += 1

    print('Verifying lengths...')
    print('All lengths equal?')

    lengths = [len(x) for x in [isbns, titles, pages, wc, batch, page_nums, levels]]
    print(lengths)

    print('Writing datasets...')
    df_dict = {'isbn': isbns, 'title': titles, 'page_text': pages, 'page_word_count': wc, 'batch': batch, 'page_num': page_nums,  'level': levels }     
    df = pd.DataFrame(df_dict)
    df.to_csv('../cs229_sp22_dataset/full_processed_dataset.csv')
    df_reduced = df[['level', 'page_text']].copy()

    df_reduced.to_csv('../cs229_sp22_dataset/level_to_page.tsv', header=None, index=(), sep='\t', mode='w')
    df_reduced.to_csv('../cs229_sp22_dataset/level_to_page.csv', header=None, index=(),  mode='w')
    print('Done.')
if __name__ == '__main__':
    make_datafiles()