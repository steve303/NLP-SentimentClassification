import emoji
def preprocess_text(input_data):
    data = input_data 

    # Removing tags
    #data = tf.strings.regex_replace(input_data, "@USER", " ")
    #data = tf.strings.regex_replace(data, "<URL>", " ")

    # Process emojis
    data = emoji.demojize(data)

    return data