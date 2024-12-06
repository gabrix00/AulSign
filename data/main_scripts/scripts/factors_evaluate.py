from itertools import zip_longest

def calculate_distances(pred_file_factor_x, gold_file_factor_x, pred_file_factor_y, gold_file_factor_y, output_path, order_factor=False):
    
    file_mode = 'a' if order_factor else 'w'

    with open(pred_file_factor_x) as file_hyps_x, open(gold_file_factor_x) as file_gold_x, \
         open(pred_file_factor_y) as file_hyps_y, open(gold_file_factor_y) as file_gold_y, \
         open(output_path,file_mode) as output:
        
        total_distance_x = 0
        total_distance_y = 0
        total_number = 0

        hyps_lines_x = [line.rstrip() for line in file_hyps_x.readlines()]
        gold_lines_x = [line.rstrip() for line in file_gold_x.readlines()]
        hyps_lines_y = [line.rstrip() for line in file_hyps_y.readlines()]
        gold_lines_y = [line.rstrip() for line in file_gold_y.readlines()]

        # Verifichiamo che tutti i file abbiano lo stesso numero di linee
        assert len(hyps_lines_x) == len(gold_lines_x) == len(hyps_lines_y) == len(gold_lines_y)

        for hyps_line_x, gold_line_x, hyps_line_y, gold_line_y in zip(hyps_lines_x, gold_lines_x, hyps_lines_y, gold_lines_y):
            hyps_numbers_x = hyps_line_x.split(' ')
            gold_numbers_x = gold_line_x.split(' ')
            hyps_numbers_y = hyps_line_y.split(' ')
            gold_numbers_y = gold_line_y.split(' ')

            zipped_x = list(zip_longest(hyps_numbers_x, gold_numbers_x, fillvalue=482))
            zipped_y = list(zip_longest(hyps_numbers_y, gold_numbers_y, fillvalue=482))

            total_number += len(zipped_x)

            for hyps_x, gold_x in zipped_x:
                distance_x = abs(int(hyps_x) - int(gold_x))
                total_distance_x += distance_x

            for hyps_y, gold_y in zipped_y:
                distance_y = abs(int(hyps_y) - int(gold_y))
                total_distance_y += distance_y

        mean_distance_x = total_distance_x / total_number
        mean_distance_y = total_distance_y / total_number

        # Assuming 'output' is a file object opened for writing
        if order_factor == True:
            output.write('======ORDERED FACTOR======\n\n')
        else:
            output.write('======NOT ORDERED FACTOR======\n\n')

        output.write('X coordinates:\n')
        output.write(f'total distance: {total_distance_x}\n')
        output.write(f'token number: {total_number}\n')
        output.write(f'mean distance: {mean_distance_x}\n')
        output.write('\nY coordinates:\n')
        output.write(f'total distance: {total_distance_y}\n')
        output.write(f'token number: {total_number}\n')
        output.write(f'mean distance: {mean_distance_y}\n\n')
