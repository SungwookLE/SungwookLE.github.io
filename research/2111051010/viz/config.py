import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='Realtime Visualize script')

    # Model arguments
    parser.add_argument('--model_multi', default='', type=str, help='model multi head name')
    parser.add_argument('--model_belt', default='', type=str, help='model belt name')
    
    # Demo setting
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')

    args = parser.parse_args()
    return args

