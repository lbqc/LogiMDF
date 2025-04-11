import torch


# Global variable to cache predicted results
predict_cache = {}


def predict_cls_name(keyword, tokenizer, encoder, ski_cluster, cls2name, predict_cache=predict_cache, max_length=24):
    """
    Predicts the category of a given keyword using the provided model.

    Args:
        keyword (str): The keyword to predict.
        tokenizer: Tokenizer used to convert text to model input.
        encoder: Encoder used to encode text into hidden state representation.
        ski_cluster: Pre-trained keyword classification model.
        cls2name (list or dict): List mapping classification labels to class names.
        max_length (int): Maximum length of input text (default is 24).

    Returns:
        str: Predicted category of the keyword or the keyword itself.

    Note:
        The function uses the global variable predict_cache to cache predicted results
        in order to avoid redundant computations.
    """
    # Check if the predicted result for the keyword is already cached
    if keyword in predict_cache:
        return predict_cache[keyword]

    # Convert the keyword to a token for model input
    keyword_token = tokenizer(
        keyword,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    # Encode the keyword using the encoder
    with torch.no_grad():
        keyword_token = keyword_token.to(encoder.device)
        _, keyword_encoded = encoder(**keyword_token, return_dict=False)

    # Predict the category using the classification model
    y_predict = ski_cluster.predict(keyword_encoded.reshape(1, -1).cpu().detach().numpy())
    keyword_cls = int(y_predict)
    
    # Cache the predicted result
    if keyword_cls < len(cls2name):
        predict_cache[keyword] = cls2name[keyword_cls]
    else:
        predict_cache[keyword] = keyword
    
    return predict_cache[keyword]


# Global variable to cache predicted results
predict_cache_global = {}


def predict_category_name(keywords, 
                          tokenizer, encoder, ski_cluster, 
                          ignore_pattern='false : ', max_length=32, encoder_output='pooler_output', 
                          cls2name=None, predict_cache=None, ):
    """
    :param keywords: The keywords to predict.
    :param tokenizer: Tokenizer used to convert text to model input.
    :param encoder: Encoder used to encode text into hidden state representation.
    :param ski_cluster: Pre-trained keyword classification model.
    :param max_length: Maximum length of input text (default is 32).
    :param encoder_output: Default is 'pooler_output', or choose 'last_hidden_state'.
    :param cls2name: (list or dict) List mapping classification labels to class names.
    :param predict_cache: Cache the predicted results.
    :return: Predicted category of the keyword or the keyword itself.
    """
    global predict_cache_global

    # Check if the predicted result for the keyword is already cached
    if predict_cache is None:
        predict_cache = predict_cache_global
    
    # print(keywords)
    keywords_old2new = {keyword: keyword.replace(ignore_pattern, '') for keyword in keywords}
        
    keywords = set(keywords_old2new.values())
    result = {}
    keywords_needed_cls = []
    for keyword in keywords:
        if keyword in predict_cache:
            result[keyword] = predict_cache[keyword]
        else:
            result[keyword] = keyword
            keywords_needed_cls.append(keyword)

    if len(keywords_needed_cls) == 0:
        for k, v in keywords_old2new.items():
            keywords_old2new[k] = result[v]
        return keywords_old2new

    # Convert the keyword to a token for model input
    # print(keywords_needed_cls)
    # print(max_length)
    keywords_token = tokenizer.batch_encode_plus(
        keywords_needed_cls,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    # Encode the keyword using the encoder
    with torch.no_grad():
        # print(encoder.device)
        keywords_token = keywords_token.to(encoder.device)
        keywords_encoded = encoder(**keywords_token)[encoder_output]

    # Predict the category using the classification model
    y_predict = ski_cluster.predict(keywords_encoded.reshape(keywords_encoded.shape[0], -1).cpu().detach().numpy())
    
    if cls2name is not None:
        y_predict_k2cname = {keywords_needed_cls[index]: cls2name[int(yi)] if yi < len(cls2name) else keywords_needed_cls[index] for index, yi in enumerate(y_predict)}
    else:
        y_predict_k2cname = {keywords_needed_cls[index]: int(yi) for index, yi in enumerate(y_predict)}

    # Cache the predicted result
    predict_cache.update(y_predict_k2cname)
    result.update(y_predict_k2cname)
    
    for k, v in keywords_old2new.items():
        keywords_old2new[k] = result[v]
    return keywords_old2new
