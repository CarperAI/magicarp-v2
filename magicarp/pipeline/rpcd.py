from typing import Tuple, Iterable, Callable, List, Any
from torchtyping import TensorType

import os
from PIL import Image
import pandas as pd
import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import einops as eo

from magicarp.pipeline import Pipeline
from magicarp.data import ImageElement, TextElement, DataElement


# ============== Data processing ==============

# Filter the dataset given ids in the following manner:
# Apply filters sequentially to save compute time
# 1. Filter out ids that do not have corresponding image files
# 2. Filter out ids that do not have corresponding comment files
# 3. Filter out ids that do not have a comment
# 4. Filter out ids that do not have at least threshold comments
def filter_id(path, id, threshold = 10):
    root = path + "/images"
    img_path = os.path.join(root, id) + "-image."

    # 1
    if not (os.path.exists(img_path + "jpg") \
        or os.path.exists(img_path + "png") \
        or os.path.exists(img_path + "jpeg")):
        return False

    # 2
    root = path + "/comments"
    years = list(range(2009, 2023))
    years = [str(year) for year in years]
    stem = "-photocritique-submission_"
    suffix = "-comments.csv"

    for year in years:
        path = os.path.join(root, year + stem + id + suffix)
        if os.path.exists(path):
            # 3. 
            try:
                df = pd.read_csv(path)
                df = df[df["comment_depth"] == 0]
                if df.empty or len(df) <= threshold:
                    del df
                    return False
                del df
                return True
            except:
                return False # Some files are bugged
    
    return False

# Given an id, load corresponding img as PIL image
def load_img(root, id):
    root += "/images"
    img_path = os.path.join(root, id) + "-image."

    if os.path.exists(img_path + "jpg"):
        img_path += "jpg"
    elif os.path.exists(img_path + "png"):
        img_path += "png"
    elif os.path.exists(img_path + "jpeg"):
        img_path += "jpeg"
    else:
        raise Exception(f"No image file for id: {id}")
    
    img = Image.open(img_path).convert("RGB")

    return img

def score_map(x : float) -> float:
    return 0.5 + 0.5 * np.tanh(0.154 * (x - 1))

# Given an  id, return list of strings corresponding to comments
# for said image
def load_comments(root, id):
    root += "/comments"

    # Comment files exist as CSV, where path is of form
    # YYYY-photocritique-submission_XXXXXX-comments.csv
    # where YYYY is the year, XXXXXX is the submission id
    # We want to find given just the submission id

    years = list(range(2009, 2023))
    years = [str(year) for year in years]
    stem = "-photocritique-submission_"
    suffix = "-comments.csv"

    for year in years:
        path = os.path.join(root, year + stem + id + suffix)
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Filter for rows where comment depth is 0
            # (i.e. top-level comments)
            df = df[df["comment_depth"] == 0]

            df = df.sort_values(by="comment_score", ascending=False)

            # return list of comments
            comments = df["comment_body"].tolist()   
            score = df["comment_score"]
            #score = score.map(lambda x : score_map(float(x)))
            score = score.tolist()

            return list(zip(comments, score))

    raise Exception(f"No comment file for id: {id}")

def weighted_sample(comments : List[Tuple[str, float]]) -> Tuple[str, float]:
    # Weighted sample that favors comments earlier in list (corresponding to higher absolute score)
    # Compute weights
    weights = [1 / (i + 1) for i in range(len(comments))]
    weights = np.array(weights)
    weights /= weights.sum()

    # Sample
    idx = np.random.choice(len(comments), p=weights)
    return comments[idx]

# =======================================
        
class RPCDPipeline(Pipeline):
    """
    Pipeline for the Reddit Photography Critique Dataset

    :param path: Path to the dataset
    :type path: str

    :param device: Device to load data onto
    :type device: torch.device

    :param force_new: Whether to force a new preprocessing for loading ids for posts/comments
    :type force_new: bool

    :param min_comments: Minimum number of comments for a post to be included in the dataset
    :type min_comments: int
    """
    def __init__(self, path : str, device : torch.device, force_new : bool = False, min_comments : int = 5):
        super().__init__()

        self.prep : Callable[[
            Iterable[Image.Image], # Photo
            Iterable[Tuple[str, float]] # Critiques and their normalized upvote counts
            ],
            Iterable[DataElement]
        ] = None

        self.root = path
        self.device = device

        paths = os.listdir(path)

        # Try to load filtered ids if they already exist
        try:
            assert not force_new
            self.ids : Iterable[int] = joblib.load(os.path.join(self.root, "aestheval_ids.joblib"))
        except:
            # Otherwise, we run the filtering process
            years = list(range(2009, 2023))
            stem = "-photocritique-submissions.csv"

            # Load all CSVs and append them
            dfs = []
            for year in years:
                df = pd.read_csv(os.path.join(path, str(year) + stem))
                dfs.append(df)
            
            df = pd.concat(dfs)

            # Validate that all ids exist in image directory
            # remove those that don't
            df = df[df["id"].apply(lambda x: filter_id(path, x, threshold = min_comments))]

            self.ids = df["id"].tolist()
            # save the filtered ids
            joblib.dump(self.ids, os.path.join(self.root, "aestheval_ids.joblib"))

        print(f"Loaded {len(self.ids)} ids after filtering")

        self.val_set : Pipeline = None

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        img = load_img(self.root, id)
        comments = load_comments(self.root, id)

        # Return top comment
        return img, weighted_sample(comments)
    
    def create_preprocess_fn(self, call_feature_extractor: Callable[[Iterable[Any]], Any]):
        def prep(batch_A : Iterable[Image.Image], batch_B : Iterable[Tuple[str, float]]) \
            -> Iterable[DataElement]:

            img_batch = batch_A
            text_batch = [txt for txt, _ in batch_B]
            scores = [score for _, score in batch_B]
            scores = torch.tensor(scores, dtype=torch.float32, device=self.device)

            img_inputs, text_inputs = call_feature_extractor(img_batch, text_batch)

            return [
                ImageElement(pixel_values=img_inputs.pixel_values).to(self.device),
                TextElement(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask).to(self.device),
                scores
            ]
        
        self.prep = prep
    
    def partition_validation_set(self, val_split : float = 0.1, shuffle : bool = False):
        if val_split is None:
            raise Exception("Validation split must be specified to use validation set")

        if shuffle:
            random.shuffle(self.ids)
        
        val_size = int(val_split * len(self.ids))
        val_ids = self.ids[:val_size]
        train_ids = self.ids[val_size:]

        self.ids = train_ids

        self.val_set = RPCDValidation(val_ids, self.root, self.device)
        self.val_set.prep = self.prep

    def create_validation_loader(self, **kwargs) -> DataLoader:
        """
        Create a dataloader for validation data. 
        """
        if self.val_set is None:
            raise Exception("Validation set not created. Call partition_validation_set() first")

        return self.val_set.create_loader(**kwargs)

class RPCDValidation(RPCDPipeline):
    def __init__(self, ids : Iterable[int], root : str, device : torch.device):
        super().__init__(root, device, force_new = False)
        self.ids = ids

# RPCD pipeline for instruct type rerankers
# Main difference is that get item returns an image and ALL comments
# on that image
# The instructGPT reward model is trained with each batch being a single
# prompt and all its responses

class InstructRPCD(RPCDPipeline):
    def __init__(self, max_comments : int = 9, **kwargs):
        super().__init__(**kwargs)

        self.max_comments = max_comments

        self.prep_fn : Callable[
            [
                Iterable[Image.Image],
                Iterable[Iterable[str]]
            ],
            Iterable[DataElement] # [ImageElement0, ImageElement0,..., TextElement0, TextElement1, ...] 
        ]

    def __getitem__(self, idx):
        id = self.ids[idx]
        img = load_img(self.root, id)
        comments = load_comments(self.root, id)
        return img, comments[:self.max_comments]

    def create_preprocess_fn(self, call_feature_extractor : Callable):
        def prep(batch_A : Iterable[Image.Image], batch_B : Iterable[Iterable[str]]) \
            -> Iterable[DataElement]:
            img_batch = batch_A
            text_batch : Iterable[str] = batch_B[0]

            img_inputs, txt_inputs = call_feature_extractor(img_batch, text_batch)

            b = len(txt_inputs.input_ids)
            # repeat the image features b times to match text batch

            return [
                ImageElement(pixel_values=img_inputs.pixel_values.repeat(b, 1, 1,1)).to(self.device),
                TextElement(input_ids=txt_inputs.input_ids, attention_mask=txt_inputs.attention_mask).to(self.device)
            ]
        
        self.prep = prep

    def partition_validation_set(self, val_split : float = 0.1, shuffle : bool = False):
        if val_split is None:
            raise Exception("Validation split must be specified to use validation set")

        if shuffle:
            random.shuffle(self.ids)
        
        val_size = int(val_split * len(self.ids))
        val_ids = self.ids[:val_size]
        train_ids = self.ids[val_size:]

        self.ids = train_ids

        self.val_set = InstructRPCDValidation(val_ids, self.root, self.device)
        self.val_set.prep = self.prep
    
    # Force loader to take batch size of 1
    def create_loader(self, **kwargs):
        # remote batch size from kwargs if its there
        kwargs.pop("batch_size", None)
        return super().create_loader(batch_size=1, **kwargs)


class InstructRPCDValidation(InstructRPCD):
    def __init__(self, ids : Iterable[int], root : str, device : torch.device):
        super().__init__(path = root, device = device, force_new = False)
        self.ids = ids

    


    

