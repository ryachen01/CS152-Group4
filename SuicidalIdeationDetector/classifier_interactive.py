from train_classifier import BertClassifier, clean_sentence
import argparse

def main():

    parser = argparse.ArgumentParser(description='Run BERT classifier with specified model')
    parser.add_argument('--model', type=str, default="bert-base-uncased_suicide_classifier",
                      help='Name of the BERT model to use (default: bert-base-uncased_suicide_classifier)')
    args = parser.parse_args()

    bert_loader = BertClassifier(model_name=args.model)
    
    bert_loader.load_model(args.model)
    # bert_loader.model.eval()

    while True:
        user_input = input("\nEnter text to classify (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        predicted_class, confidence = bert_loader.predict_text(user_input)
        print(f"\nPrediction for text: '{clean_sentence(user_input)}'")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
    

if __name__ == "__main__":
    main()
    