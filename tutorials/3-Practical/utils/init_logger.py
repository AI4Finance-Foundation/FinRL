def init_logger(log_path):
    import logging
    logger = logging.getLogger(log_path)
    logger.propagate = False
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s,%(msecs)d (%(name)s) [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logging.root = logger


def logger_test(log_path):
    init_logger(log_path)

    import logging

    for _ in range(10):
        logging.info('test')


if __name__ == '__main__':
    import multiprocessing as mp
    import time

    mp.freeze_support()
    n_process = 2
    process_list = [mp.Process(target=logger_test, kwargs={'log_path': 'test{}.log'.format(i)}) for i in
                    range(n_process)]
    for process in process_list:
        process.daemon = True
        process.start()

    time.sleep(1)
