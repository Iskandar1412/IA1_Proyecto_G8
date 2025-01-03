import React from 'react';
import { SideBar } from '../../components/layout';
import { NavBar } from '../../components/layout/NavBar';

export const Layout = ({ children }) => {
  return (
    <div className="bg-slate-100 overflow-y-scroll w-screen h-screen antialiased text-slate-300 selection:bg-blue-600 selection:text-white">
      {/* <NavBar /> */}
      <div className="flex">
        <SideBar />
        <div className="p-2 w-full text-slate-900">{children}</div>
      </div>
    </div>
  );
};
