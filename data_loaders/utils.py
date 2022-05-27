from datetime import timedelta
from conflab.constants import vid2_start, vid3_start, vid_timecodes

def time_to_seg(time, sr=59.94):
    '''
    Converts a time (s) in the annotated section of the dataset 
    into a segment number and offset that can be used to localize
    a window within the loaded tracks
    '''
    vid2_seg8_start = vid2_start + timedelta(minutes=14)
    vid2_len = (vid3_start - vid2_seg8_start).total_seconds()
    if time < vid2_len:
        seg = int(time // 120) # 2 min segments
        offset = int((time % 120) * sr)
    else:
        seg = 2+int((time - vid2_len) // 120)
        offset = int(((time - vid2_len) % 120) * sr)

    return seg, offset

def seg_to_offset(seg):
    if seg <= 1:
        return seg*120
    else:
        vid2_seg8_start = vid2_start + timedelta(minutes=14)
        vid2_len = (vid3_start - vid2_seg8_start).total_seconds()
        return vid2_len + (seg-2)*120


def vid_seg_to_segment(vid, seg):
    return {
        (2,8): 0,
        (2,9): 1,
        (3,1): 2,
        (3,2): 3,
        (3,3): 4,
        (3,4): 5,
        (3,5): 6,
        (3,6): 7
    }[(vid,seg)]