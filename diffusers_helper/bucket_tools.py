bucket_options = {
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    if resolution not in bucket_options: bucket_options[resolution] = [(max(16, round(h * resolution / 640 / 16) * 16), max(16, round(w * resolution / 640 / 16) * 16)) for h, w in bucket_options[640]]
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

