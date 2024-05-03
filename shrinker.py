import csv


def main():
    input_file = '/Users/timoniche/Documents/BigData/BigDataLab5/en.openfoodfacts.org.products.csv'
    output_file = '/Users/timoniche/Documents/BigData/BigDataLab5/small.csv'

    with open(input_file, 'r') as f_input:
        with open(output_file, 'w') as f_output:
            csv_input = csv.reader(f_input)
            csv_output = csv.writer(f_output)

            header = next(csv_input)
            csv_output.writerow(header)

            for _ in range(1000):
                try:
                    row = next(csv_input)
                    csv_output.writerow(row)
                except StopIteration:
                    break


if __name__ == '__main__':
    main()
