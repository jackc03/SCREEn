from pathlib import Path
from triplet_dataset import TripletDataset

lr = Path("datasets") / "davis_trainval_2017_720p" / "DAVIS" / "JPEGImages" / "Full-Resolution"
hr = Path("datasets") / "davis_trainval_2017_1080p" / "DAVIS" / "JPEGImages" / "Full-Resolution"

print("LR path:", lr)
print("Exists? ", lr.exists())
print("HR path:", hr)
print("Exists? ", hr.exists())

ds = TripletDataset(lr, hr, split="train")
print("Train samples:", len(ds))
for i in range(min(3, len(ds))):
    print([t.shape for t in ds[i]])
