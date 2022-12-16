"""
This file is used to run all the .py files in one instance.
"""

import hw1_206230021_q1_train
import hw1_206230021_q1_eval
import hw1_206230021_q2_train
import hw1_206230021_q2_eval


def main():
    print('\nhw1_206230021_q1_train.py\n')
    hw1_206230021_q1_train.main()
    print('\nhw1_206230021_q1_eval.py\n')
    hw1_206230021_q1_eval.main()
    print('\nhw1_206230021_q2_train.py\n')
    hw1_206230021_q2_train.main()
    print('\nhw1_206230021_q2_eval.py\n')
    hw1_206230021_q2_eval.main()


if __name__ == '__main__':
    main()
