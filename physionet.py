from torchtime.data import PhysioNet2012
import torch
from torch.utils.data import DataLoader
from multiprocessing import Manager
import pytorch_lightning as pl

#     :0. Mins: Minutes since ICU admission. Derived from the PhysioNet time stamp.
#     :1. Albumin: Albumin (g/dL)
#     :2. ALP: Alkaline phosphatase (IU/L)
#     :3. ALT: Alanine transaminase (IU/L)
#     :4. AST: Aspartate transaminase (IU/L)
#     :5. Bilirubin: Bilirubin (mg/dL)
#     :6. BUN: Blood urea nitrogen (mg/dL)
#     :7. Cholesterol: Cholesterol (mg/dL)
#     :8. Creatinine: Serum creatinine (mg/dL)
#     :9. DiasABP: Invasive diastolic arterial blood pressure (mmHg)
#     :10. FiO2: Fractional inspired O\ :sub:`2` (0-1)
#     :11. GCS: Glasgow Coma Score (3-15)
#     :12. Glucose: Serum glucose (mg/dL)
#     :13. HCO3: Serum bicarbonate (mmol/L)
#     :14. HCT: Hematocrit (%)
#     :15. HR: Heart rate (bpm)
#     :16. K: Serum potassium (mEq/L)
#     :17. Lactate: Lactate (mmol/L)
#     :18. Mg: Serum magnesium (mmol/L)
#     :19. MAP: Invasive mean arterial blood pressure (mmHg)
#     :20. MechVent: Mechanical ventilation respiration (0:false, or 1:true)
#     :21. Na: Serum sodium (mEq/L)
#     :22. NIDiasABP: Non-invasive diastolic arterial blood pressure (mmHg)
#     :23. NIMAP: Non-invasive mean arterial blood pressure (mmHg)
#     :24. NISysABP: Non-invasive systolic arterial blood pressure (mmHg)
#     :25. PaCO2: Partial pressure of arterial CO\ :sub:`2` (mmHg)]
#     :26. PaO2: Partial pressure of arterial O\ :sub:`2` (mmHg)
#     :27. pH: Arterial pH (0-14)
#     :28. Platelets: Platelets (cells/nL)
#     :29. RespRate: Respiration rate (bpm)
#     :30. SaO2: O\ :sub:`2` saturation in hemoglobin (%)
#     :31. SysABP: Invasive systolic arterial blood pressure (mmHg)
#     :32. Temp: Temperature (°C)
#     :33. TroponinI: Troponin-I (μg/L). Note this is labelled *TropI* in the PhysioNet
#         data dictionary.
#     :34. TroponinT: Troponin-T (μg/L). Note this is labelled *TropT* in the PhysioNet
#         data dictionary.
#     :35. Urine: Urine output (mL)
#     :36. WBC: White blood cell count (cells/nL)
#     :37. Weight: Weight (kg)
#     :38. Age: Age (years) at ICU admission
#     :39. Gender: Gender (0: female, or 1: male)
#     :40. Height: Height (cm) at ICU admission
#     :41. ICUType1: Type of ICU unit (1: Coronary Care Unit)
#     :42. ICUType2: Type of ICU unit (2: Cardiac Surgery Recovery Unit)
#     :43. ICUType3: Type of ICU unit (3: Medical ICU)
#     :44. ICUType4: Type of ICU unit (4: Surgical ICU)

class PhysioNetDataset(torch.utils.data.Dataset):
    def __init__(self, split_name, n_timesteps=32, use_temp_cache=False, **kwargs):
        self.split_name = split_name
        self.n_timesteps = n_timesteps
        self.temp_cache = Manager().dict() if use_temp_cache else None

    def setup(self):
        # To maintain consistent splits, we use a seed of 0 here regardless of the model initialization seed
        tt_data = PhysioNet2012(self.split_name, train_prop=0.7, val_prop=0.15, time=False, seed=0)

        self.X = tt_data.X
        self.y = tt_data.y

        self.means = []
        self.stds = []
        self.maxes = []
        self.mins = []
        for i in range(self.X.shape[2]):
            vals = self.X[:,:,i].flatten()
            vals = vals[~torch.isnan(vals)]
            self.means.append(vals.mean())
            self.stds.append(vals.std())
            self.maxes.append(vals.max())
            self.mins.append(vals.min())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.temp_cache is not None and i in self.temp_cache:
            return self.temp_cache[i]

        ins = self.X[i, ~torch.isnan(self.X[i,:,0]), :]
        time = ins[:,0] / 60 / 24
        x_static = torch.zeros(self.d_static_num())

        x_ts = torch.zeros((self.n_timesteps, self.d_time_series_num()*2))
        for i_t, t in enumerate(time):
            bin = self.n_timesteps - 1 if t == time[-1] else int(t / time[-1] * self.n_timesteps)
            for i_ts in range(1,37):
                x_i = ins[i_t,i_ts]
                if not torch.isnan(x_i).item():
                    x_ts[bin, i_ts-1] = (x_i - self.means[i_ts])/(self.stds[i_ts] + 1e-7)
                    x_ts[bin, i_ts-1+self.d_time_series_num()] += 1
        bin_ends = torch.arange(1, self.n_timesteps+1) / self.n_timesteps * time[-1]

        for i_tab in range(37,45):
            x_i = ins[0, i_tab]
            x_i = (x_i - self.means[i_tab])/(self.stds[i_tab] + 1e-7)
            x_static[i_tab-37] = x_i.nan_to_num(0.)

        x = (x_ts, x_static, bin_ends)
        y = self.y[i,0]
        if self.temp_cache is not None:
            self.temp_cache[i] = (x, y)

        return x, y

    def d_static_num(self):
        """The total dimension of numeric static features"""
        return 8

    def d_time_series_num(self):
        """The total dimension of numeric time-series features"""
        return 36

    def d_target(self):
        return 1

    def pos_frac(self):
        return self.y.mean().numpy()

def collate_into_seqs(batch):
    xs, ys = zip(*batch)
    return zip(*xs), ys

class PhysioNetDataModule(pl.LightningDataModule):
    def __init__(self, use_temp_cache=False, batch_size=8, num_workers=1, prefetch_factor=2,
            verbose=0, **kwargs):
        self.use_temp_cache = use_temp_cache
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.ds_train = PhysioNetDataset('train', use_temp_cache=use_temp_cache)
        self.ds_val = PhysioNetDataset('val', use_temp_cache=use_temp_cache)
        self.ds_test = PhysioNetDataset('test', use_temp_cache=use_temp_cache)

        self.prepare_data_per_node = False

        self.dl_args = {'batch_size': self.batch_size, 'prefetch_factor': self.prefetch_factor,
                'collate_fn': collate_into_seqs, 'num_workers': num_workers}

    def setup(self, stage=None):
        if stage is None:
            self.ds_train.setup()
            self.ds_val.setup()
            self.ds_test.setup()
        elif stage == 'fit':
            self.ds_train.setup()
            self.ds_val.setup()
        elif stage == 'validate':
            self.ds_val.setup()
        elif stage == 'test':
            self.ds_test.setup()

    def prepare_data(self):
        pass

    def _log_hyperparams(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.ds_train, shuffle=True, **self.dl_args)

    def val_dataloader(self):
        return DataLoader(self.ds_val, **self.dl_args)

    def test_dataloader(self):
        return DataLoader(self.ds_test, **self.dl_args)

    def d_static_num(self):
        return self.ds_train.d_static_num()

    def d_time_series_num(self):
        return self.ds_train.d_time_series_num()

    def d_target(self):
        return self.ds_train.d_target()

    def pos_frac(self):
        return self.ds_train.pos_frac()
