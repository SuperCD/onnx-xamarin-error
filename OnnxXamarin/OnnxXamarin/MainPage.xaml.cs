﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Forms;

namespace OnnxXamarin
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            OnnxLoader.LoadModel();
            InitializeComponent();
        }
    }
}

