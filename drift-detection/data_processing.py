import numpy as np
import pandas as pd
import os
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f'Initial memory usage: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage decreased to {:.2f} MB ({:.1f}% reduction)'.format(
        end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def load_data(data_dir):
    logger.info(f"Loading data from {data_dir}...")
    try:
        train_id = pd.read_csv(os.path.join(data_dir, 'train_identity.csv'))
        train_trn = pd.read_csv(os.path.join(data_dir, 'train_transaction.csv'))
        logger.info(f"Loaded training data: {train_trn.shape}, {train_id.shape}")
        
        try:
            test_id = pd.read_csv(os.path.join(data_dir, 'test_identity.csv'))
            test_trn = pd.read_csv(os.path.join(data_dir, 'test_transaction.csv'))
            logger.info(f"Loaded test data: {test_trn.shape}, {test_id.shape}")
        except FileNotFoundError:
            test_id, test_trn = None, None
            logger.warning("Test files not found. Proceeding with train files only.")
            
        return train_id, train_trn, test_id, test_trn
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def merge_and_clean(train_trn, train_id, test_trn=None, test_id=None):
    logger.info("Merging and cleaning data...")
    id_cols = list(train_id.columns.values)
    trn_cols = list(train_trn.drop('isFraud', axis=1).columns.values)

    X_train = pd.merge(train_trn[trn_cols + ['isFraud']], train_id[id_cols], how='left')
    X_train = reduce_mem_usage(X_train)
    
    X_test = None
    if test_trn is not None and test_id is not None:
        X_test = pd.merge(test_trn[trn_cols], test_id[id_cols], how='left')
        X_test = reduce_mem_usage(X_test)

    # Clear memory
    del train_id, train_trn
    if test_id is not None: del test_id, test_trn
    gc.collect()

    return X_train, X_test
