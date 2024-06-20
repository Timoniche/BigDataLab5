import csv
import pathlib

N_ROWS = 1000


def main():
    curdir = str(pathlib.Path(__file__).parent)
    input_file = curdir + '/' + 'en.openfoodfacts.org.products.csv'
    output_file = curdir + '/' + 'dataset.csv'

    with open(input_file, 'r') as f_input:
        with open(output_file, 'w') as f_output:
            csv_input = csv.reader(f_input)
            csv_output = csv.writer(f_output)

            header = next(csv_input)
            csv_output.writerow(header)

            for _ in range(N_ROWS):
                try:
                    row = next(csv_input)
                    csv_output.writerow(row)
                except StopIteration:
                    break


if __name__ == '__main__':
    main()
