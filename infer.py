import argparse

def main(model_path: str, input_path:str) -> None:
    print("model_path", model_path)
    print("input_path", input_path)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='PytorchModelInference',
                    description='Given an input, and a pytorch model, this program provides output of the model.',
                    )
    parser.add_argument('-m', '--model_path', type=str, default="",
                        help='The path to the saved model (jit version only)')
    parser.add_argument('-i', '--input_audio', type=str, default="",
                        help='The path to the input audio file')

    args = parser.parse_args()
    main(args.model_path, args.input_audio)



