import React, { ReactNode } from 'react';
import clsx from 'clsx';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
  gradient?: boolean;
  icon?: ReactNode;
}

export function Card({ title, children, className, action, gradient = false, icon }: CardProps) {
  return (
    <div
      className={clsx(
        'bg-white rounded-xl shadow-sm border border-gray-200 transition-all duration-200 hover:shadow-md',
        className
      )}
    >
      {title && (
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {icon && <div className="text-gray-500">{icon}</div>}
            <h3 className="text-lg font-bold text-slate-900">
              {title}
            </h3>
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
}

