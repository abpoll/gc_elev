# -*- coding: utf-8 -*-


def main(input_filepath, output_filepath):
    """ Runs data downloading scripts to get data hosted on webservers
        and store them in raw/
    """
    logger = logging.getLogger(__name__)
    logger.info('downloading raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
